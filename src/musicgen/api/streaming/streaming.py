"""
WebSocket-based streaming API for real-time music generation updates.
Provides live progress, partial results, and generation metrics.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

# Type checking imports (only for IDE/mypy, not runtime)
if TYPE_CHECKING:
    from ...core.generator import MusicGenerator
    from ...core.prompt import PromptEngineer
else:
    MusicGenerator = None
    PromptEngineer = None


# Lazy imports for heavy ML dependencies
def get_music_generator():
    """Lazy import MusicGenerator."""
    from ...core.generator import MusicGenerator

    return MusicGenerator


def get_prompt_engineer():
    """Lazy import PromptEngineer."""
    from ...core.prompt import PromptEngineer

    return PromptEngineer


class StreamingSession:
    """Manages a single streaming generation session."""

    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.generator = None  # Optional[MusicGenerator]
        self.is_active = True
        self.start_time = time.time()
        self.generation_task: Optional[asyncio.Task] = None

    async def send_message(self, message_type: str, data: Dict[str, Any]):
        """Send a message to the client with error handling."""
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_json(
                    {
                        "type": message_type,
                        "timestamp": datetime.now().isoformat(),
                        "session_id": self.session_id,
                        "data": data,
                    }
                )
        except Exception as e:
            # Connection lost, mark session as inactive
            self.is_active = False
            raise WebSocketDisconnect()

    async def close(self):
        """Close the session and cleanup resources."""
        self.is_active = False

        # Cancel any running generation task
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass

        # Cleanup generator resources
        if self.generator:
            # Clear GPU memory if using CUDA
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            self.generator = None

        # Close WebSocket if still connected
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()


class StreamingManager:
    """Manages all active streaming sessions."""

    def __init__(self):
        self.sessions: Dict[str, StreamingSession] = {}
        self._generator_cache = None  # Optional[MusicGenerator]

    def get_generator(self):
        """Get or create a shared generator instance."""
        if self._generator_cache is None:
            MusicGenerator = get_music_generator()
            self._generator_cache = MusicGenerator()
        return self._generator_cache

    async def create_session(self, websocket: WebSocket) -> StreamingSession:
        """Create a new streaming session."""
        await websocket.accept()
        session_id = str(uuid.uuid4())
        session = StreamingSession(session_id, websocket)
        self.sessions[session_id] = session

        # Send initial connection message
        await session.send_message(
            "connected",
            {
                "session_id": session_id,
                "status": "ready",
                "capabilities": {
                    "max_duration": 300,
                    "formats": ["wav", "mp3"],
                    "streaming": True,
                    "partial_results": True,
                },
            },
        )

        return session

    async def remove_session(self, session_id: str):
        """Remove and cleanup a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            await session.close()
            del self.sessions[session_id]

            # If no more sessions and generator exists, consider cleanup
            if not self.sessions and self._generator_cache:
                # Clear GPU memory
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    async def handle_generation_request(
        self,
        session: StreamingSession,
        prompt: str,
        duration: float = 30.0,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        **kwargs,
    ):
        """Handle a music generation request with streaming updates."""
        try:
            # Validate prompt
            PromptEngineer = get_prompt_engineer()
            engineer = PromptEngineer()
            is_valid, issues = engineer.validate_prompt(prompt)

            if not is_valid:
                await session.send_message(
                    "validation_error",
                    {"issues": issues, "suggestion": engineer.improve_prompt(prompt)},
                )
                return

            # Send generation started message
            await session.send_message(
                "generation_started",
                {
                    "prompt": prompt,
                    "duration": duration,
                    "parameters": {"temperature": temperature, "guidance_scale": guidance_scale},
                },
            )

            # Get generator
            generator = self.get_generator()
            session.generator = generator

            # Define progress callback
            async def progress_callback(percent: float, message: str):
                await session.send_message(
                    "progress",
                    {
                        "percent": percent,
                        "message": message,
                        "elapsed_time": time.time() - session.start_time,
                    },
                )

            # Create generation task that can be cancelled
            generation_coro = None
            if duration <= 30:
                # Single generation
                generation_coro = self._generate_single(
                    session,
                    generator,
                    prompt,
                    duration,
                    temperature,
                    guidance_scale,
                    progress_callback,
                )
            else:
                # Extended generation with segments
                generation_coro = self._generate_extended(
                    session,
                    generator,
                    prompt,
                    duration,
                    temperature,
                    guidance_scale,
                    progress_callback,
                )

            # Store task for potential cancellation
            session.generation_task = asyncio.create_task(generation_coro)
            await session.generation_task

        except asyncio.CancelledError:
            await session.send_message("generation_cancelled", {"reason": "User cancelled"})
        except Exception as e:
            await session.send_message("error", {"error": str(e), "type": type(e).__name__})

    async def _generate_single(
        self,
        session: StreamingSession,
        generator,  # MusicGenerator instance
        prompt: str,
        duration: float,
        temperature: float,
        guidance_scale: float,
        progress_callback,
    ):
        """Generate a single audio segment."""
        # Run generation in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def sync_generate():
            # Wrapper for sync generation with progress
            progress_data = {"percent": 0, "message": ""}

            def sync_progress(percent, message):
                progress_data["percent"] = percent
                progress_data["message"] = message

            audio, sample_rate = generator.generate(
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                guidance_scale=guidance_scale,
                progress_callback=sync_progress,
            )
            return audio, sample_rate, progress_data

        # Generate with periodic progress updates
        generation_future = loop.run_in_executor(None, sync_generate)

        # Send progress updates while generating
        while not generation_future.done():
            await asyncio.sleep(0.5)
            # Check if we have progress updates to send

        audio, sample_rate, final_progress = await generation_future

        # Send completion message with audio metadata
        await session.send_message(
            "generation_complete",
            {
                "audio_shape": audio.shape,
                "sample_rate": sample_rate,
                "duration": len(audio) / sample_rate,
                "format": "float32",
                "generation_time": time.time() - session.start_time,
            },
        )

        # Optionally send audio chunks for streaming playback
        await self._stream_audio_chunks(session, audio, sample_rate)

    async def _generate_extended(
        self,
        session: StreamingSession,
        generator,  # MusicGenerator instance
        prompt: str,
        duration: float,
        temperature: float,
        guidance_scale: float,
        progress_callback,
    ):
        """Generate extended audio with multiple segments."""
        segment_duration = 25.0
        overlap = 2.0
        num_segments = int(np.ceil(duration / (segment_duration - overlap)))

        await session.send_message(
            "extended_generation",
            {
                "total_segments": num_segments,
                "segment_duration": segment_duration,
                "overlap": overlap,
            },
        )

        segments = []
        for i in range(num_segments):
            # Send segment start message
            await session.send_message("segment_started", {"segment": i + 1, "total": num_segments})

            # Generate segment
            seg_duration = min(segment_duration, duration - i * (segment_duration - overlap))

            loop = asyncio.get_event_loop()
            audio, sample_rate = await loop.run_in_executor(
                None,
                generator.generate,
                prompt,
                seg_duration,
                temperature,
                guidance_scale,
                None,  # No progress callback for segments
            )

            segments.append(audio)

            # Send segment complete message
            await session.send_message(
                "segment_complete",
                {"segment": i + 1, "audio_shape": audio.shape, "sample_rate": sample_rate},
            )

            # Stream this segment's audio
            await self._stream_audio_chunks(session, audio, sample_rate, segment_id=i)

        # Blend segments
        await session.send_message("blending_segments", {"num_segments": len(segments)})

        loop = asyncio.get_event_loop()
        final_audio = await loop.run_in_executor(
            None, generator._blend_segments, segments, sample_rate, overlap
        )

        # Send final completion
        await session.send_message(
            "generation_complete",
            {
                "audio_shape": final_audio.shape,
                "sample_rate": sample_rate,
                "duration": len(final_audio) / sample_rate,
                "num_segments": num_segments,
                "generation_time": time.time() - session.start_time,
            },
        )

    async def _stream_audio_chunks(
        self,
        session: StreamingSession,
        audio: np.ndarray,
        sample_rate: int,
        chunk_size: int = 8192,
        segment_id: Optional[int] = None,
    ):
        """Stream audio data in chunks for progressive playback."""
        num_chunks = int(np.ceil(len(audio) / chunk_size))

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio))
            chunk = audio[start_idx:end_idx]

            # Convert to base64 for transmission
            import base64

            chunk_bytes = chunk.astype(np.float32).tobytes()
            chunk_b64 = base64.b64encode(chunk_bytes).decode("utf-8")

            await session.send_message(
                "audio_chunk",
                {
                    "chunk_id": i,
                    "total_chunks": num_chunks,
                    "segment_id": segment_id,
                    "data": chunk_b64,
                    "start_sample": start_idx,
                    "end_sample": end_idx,
                    "sample_rate": sample_rate,
                },
            )

            # Small delay to avoid overwhelming the client
            await asyncio.sleep(0.01)


# Global streaming manager instance
streaming_manager = StreamingManager()


async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for streaming generation with timeout and cleanup."""
    session = None
    heartbeat_task = None

    async def heartbeat():
        """Send periodic heartbeat to detect disconnections."""
        while session and session.is_active:
            try:
                await session.send_message("heartbeat", {"timestamp": time.time()})
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except (WebSocketDisconnect, Exception):
                break

    try:
        # Create session
        session = await streaming_manager.create_session(websocket)

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat())

        # Handle messages with timeout
        while session.is_active:
            try:
                # Wait for message with 60 second timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
            except asyncio.TimeoutError:
                # Send keepalive and continue
                await session.send_message("keepalive", {})
                continue

            message_type = data.get("type")

            if message_type == "generate":
                # Start generation
                params = data.get("params", {})
                asyncio.create_task(
                    streaming_manager.handle_generation_request(
                        session,
                        prompt=params.get("prompt"),
                        duration=params.get("duration", 30.0),
                        temperature=params.get("temperature", 1.0),
                        guidance_scale=params.get("guidance_scale", 3.0),
                    )
                )

            elif message_type == "cancel":
                # Cancel current generation
                if session.generation_task and not session.generation_task.done():
                    session.generation_task.cancel()
                    try:
                        await session.generation_task
                    except asyncio.CancelledError:
                        pass
                    await session.send_message("cancelled", {"status": "Generation cancelled"})

            elif message_type == "ping":
                # Heartbeat
                await session.send_message("pong", {"timestamp": time.time()})

            elif message_type == "disconnect":
                break

    except WebSocketDisconnect:
        # Client disconnected
        pass
    except Exception as e:
        # Try to send error message
        if session:
            try:
                await session.send_message("error", {"error": str(e)})
            except:
                pass
    finally:
        # Cancel heartbeat
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # Remove and cleanup session
        if session:
            await streaming_manager.remove_session(session.session_id)


def list_sessions() -> List[Dict[str, Any]]:
    """List all active streaming sessions."""
    return [
        {
            "session_id": session_id,
            "active": session.is_active,
            "duration": time.time() - session.start_time,
        }
        for session_id, session in streaming_manager.sessions.items()
    ]
