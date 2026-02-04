# -*- coding: utf-8 -*-
# Project : Fast-Spark-TTS
# Time    : 2025/4/7 16:14
# Author  : Hui Huang
import os
from pathlib import Path
from typing import Optional, Annotated, Literal
from fastapi import (
    HTTPException,
    Request,
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends,
    BackgroundTasks,
)
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
from .protocol import CloneRequest, SpeakRequest, MultiSpeakRequest, StateInfo
from .utils.audio_writer import StreamingAudioWriter
from .utils.utils import (
    load_audio_bytes,
    load_latent_file,
    generate_audio_stream,
    generate_audio,
)
from ..engine import AutoEngine
from ..logger import get_logger

logger = get_logger()

base_router = APIRouter(
    tags=["Flash-TTS"], responses={404: {"description": "Not found"}}
)

TEMPLATES_DIR = Path(__file__).parent / "templates"

SPEAKER_TMP_PATH = "SPEAKER.bin.tmp"


def get_engine(request: Request) -> AutoEngine:
    return request.app.state.engine


def get_state_info(request: Request) -> StateInfo:
    return request.app.state.state_info


def _save_to_disk_sync(engine: AutoEngine, db_path):
    def task():
        engine.save_speakers(SPEAKER_TMP_PATH)
        os.replace(SPEAKER_TMP_PATH, db_path)

    return task


@base_router.get("/")
async def get_web():
    html_path = TEMPLATES_DIR / "index.html"
    return FileResponse(str(html_path))


@base_router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = TEMPLATES_DIR / "favicon.ico"
    return FileResponse(str(favicon_path))


@base_router.get("/server_info")
async def audio_roles(raw_request: Request):
    engine = get_engine(raw_request)
    return JSONResponse(
        content={
            "success": True,
            "info": {
                "model": engine.engine_name,
                "roles": engine.list_speakers(),
                "sample_rate": engine.SAMPLE_RATE,
            },
        }
    )


@base_router.post("/add_speaker")
async def add_speaker(
    raw_request: Request,
    background_tasks: BackgroundTasks,
    name: str = Form(..., description="The name of the speaker"),
    audio: Optional[str] = Form(
        None,
        description="A reference audio sample of the speaker (URL or base64 string). Use this or `audio_file`",
    ),
    reference_text: Optional[str] = Form(
        None,
        description="Optional transcript or description corresponding to the reference audio.",
    ),
    audio_file: Optional[UploadFile] = File(
        None,
        description="Upload reference audio file (WAV) of the speaker. Use this or `audio`",
    ),
    latent_file: Optional[UploadFile] = File(
        None, description="latent file for mega-tts."
    ),
):
    engine = get_engine(raw_request)
    state_info = get_state_info(raw_request)
    if engine.engine_name == "orpheus":
        logger.error(
            "OrpheusTTS does not currently support adding custom voice characters."
        )
        raise HTTPException(
            status_code=500,
            detail="OrpheusTTS does not currently support adding custom voice characters.",
        )

    bytes_io = await load_audio_bytes(audio_file=audio_file, audio=audio)

    if engine.engine_name == "mega":
        latent_io = await load_latent_file(latent_file=latent_file)
        reference_audio = (bytes_io, latent_io)
    else:
        reference_audio = bytes_io

    try:
        await engine.add_speaker(
            name=name, audio=reference_audio, reference_text=reference_text
        )
    except Exception as e:
        try:
            await engine.delete_speaker(name=name)
        except:
            pass
        err_msg = f'Failed to add the voice character "{name}": {str(e)}'
        logger.error(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)
    background_tasks.add_task(
        _save_to_disk_sync(engine=engine, db_path=state_info.db_path)
    )
    return JSONResponse(content={"success": True, "role": name})


@base_router.post("/delete_speaker")
async def delete_speaker(
    raw_request: Request,
    background_tasks: BackgroundTasks,
    name: str = Form(..., description="The name of the speaker"),
):
    engine = get_engine(raw_request)
    state_info = get_state_info(raw_request)
    if engine.engine_name == "orpheus":
        logger.error(
            "OrpheusTTS does not currently support deleting custom voice characters."
        )
        raise HTTPException(
            status_code=500,
            detail="OrpheusTTS does not currently support deleting custom voice characters.",
        )
    try:
        await engine.delete_speaker(name=name)
    except Exception as e:
        err_msg = f'Failed to remove the voice character "{name}": {str(e)}'
        logger.error(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)
    background_tasks.add_task(
        _save_to_disk_sync(engine=engine, db_path=state_info.db_path)
    )
    return JSONResponse(content={"success": True, "role": name})


def parse_clone_form(
    text: str = Form(...),
    reference_audio: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Form(
        None
    ),
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Form(
        None
    ),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    repetition_penalty: float = Form(1.0),
    max_tokens: int = Form(4096),
    length_threshold: int = Form(50),
    window_size: int = Form(50),
    stream: bool = Form(False),
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Form("mp3"),
):
    return CloneRequest(
        text=text,
        reference_audio=reference_audio,
        reference_text=reference_text,
        pitch=pitch,
        speed=speed,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        length_threshold=length_threshold,
        window_size=window_size,
        stream=stream,
        response_format=response_format,
    )


# 克隆语音接口：接收 multipart/form-data，上传参考音频和其它表单参数
@base_router.post("/clone_voice")
async def clone_voice(
    req: Annotated[CloneRequest, Depends(parse_clone_form)],
    raw_request: Request,
    reference_audio_file: Optional[UploadFile] = File(None),
    latent_file: Optional[UploadFile] = File(None),
):
    engine = get_engine(raw_request)
    if engine.engine_name == "orpheus":
        logger.error("OrpheusTTS does not currently support voice cloning.")
        raise HTTPException(
            status_code=500,
            detail="OrpheusTTS does not currently support voice cloning.",
        )

    bytes_io = await load_audio_bytes(
        audio_file=reference_audio_file, audio=req.reference_audio
    )

    if engine.engine_name == "mega":
        latent_io = await load_latent_file(latent_file=latent_file)
        reference_audio = (bytes_io, latent_io)
    else:
        reference_audio = bytes_io

    audio_writer = StreamingAudioWriter(
        req.response_format, sample_rate=engine.SAMPLE_RATE
    )
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    if req.stream:
        data = dict(
            text=req.text,
            reference_audio=reference_audio,
            reference_text=req.reference_text,
            pitch=req.pitch,
            speed=req.speed,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_tokens=req.max_tokens,
            length_threshold=req.length_threshold,
            window_size=req.window_size,
        )
        return StreamingResponse(
            generate_audio_stream(
                engine.clone_voice_stream_async, data, audio_writer, raw_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        try:
            audio = await engine.clone_voice_async(
                text=req.text,
                reference_audio=reference_audio,
                reference_text=req.reference_text,
                pitch=req.pitch,
                speed=req.speed,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
                length_threshold=req.length_threshold,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"Failed to clone voice: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )


@base_router.get("/audio_roles")
async def audio_roles(raw_request: Request):
    roles = raw_request.app.state.engine.list_speakers()
    return JSONResponse(content={"success": True, "roles": roles})


@base_router.post("/speak")
async def speak(req: SpeakRequest, raw_request: Request):
    engine = get_engine(raw_request)
    state_info = get_state_info(raw_request)
    if req.name not in engine.list_speakers():
        err_msg = f'"{req.name}" is not in the list of existing roles: {", ".join(engine.list_speakers())}'
        logger.warning(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)

    audio_writer = StreamingAudioWriter(
        req.response_format, sample_rate=engine.SAMPLE_RATE
    )
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    data = dict(
        name=req.name,
        text=req.text,
        temperature=req.temperature,
        pitch=req.pitch,
        speed=req.speed,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        max_tokens=req.max_tokens,
        length_threshold=req.length_threshold,
        window_size=req.window_size,
    )
    # 判断是否需要保存音色
    if state_info.fix_voice and engine.engine_name == "spark":
        req.name = req.name or "female"
        if state_info.acoustic_tokens is None:
            state_info.init_acoustic_tokens()
        if req.name in ["female", "male"]:
            # 只有这两个内置角色需要持久化音色
            if state_info.acoustic_tokens[req.name] is None:
                data["return_acoustic_tokens"] = True
            else:
                data["return_acoustic_tokens"] = False
                data["acoustic_tokens"] = state_info.acoustic_tokens[req.name]

    if req.stream:
        return StreamingResponse(
            generate_audio_stream(
                engine.speak_stream_async,
                data,
                audio_writer,
                raw_request,
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        try:
            audio = await engine.speak_async(**data)
            if isinstance(audio, tuple):
                if state_info.acoustic_tokens is not None:
                    state_info.acoustic_tokens[req.name] = audio[1]
                audio = audio[0]
        except Exception as e:
            logger.warning(f"Voice synthesis for the role failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )


@base_router.post("/multi_speak")
async def multi_speak(req: MultiSpeakRequest, raw_request: Request):
    engine = get_engine(raw_request)

    audio_writer = StreamingAudioWriter(
        req.response_format, sample_rate=engine.SAMPLE_RATE
    )
    # Set content type based on format
    content_type = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(req.response_format, f"audio/{req.response_format}")

    if req.stream:
        data = dict(
            text=req.text,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_tokens=req.max_tokens,
            length_threshold=req.length_threshold,
            window_size=req.window_size,
        )
        return StreamingResponse(
            generate_audio_stream(
                engine.multi_speak_stream_async, data, audio_writer, raw_request
            ),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        try:
            audio = await engine.multi_speak_async(
                text=req.text,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
                length_threshold=req.length_threshold,
                window_size=req.window_size,
            )
        except Exception as e:
            logger.warning(f"Multi-role voice synthesis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        headers = {
            "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
            "Cache-Control": "no-cache",  # Prevent caching
        }
        audio_io = generate_audio(audio, writer=audio_writer)
        return Response(
            audio_io,
            media_type=content_type,
            headers=headers,
        )
