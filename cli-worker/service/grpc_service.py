"""
gRPC service implementation for CLI Worker.
Dispatches RPCs to command builders and runs them in the sandbox.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import uuid
from typing import Optional

import grpc
from cli_service_pb2 import (
    BurnSubtitlesRequest,
    CliBytesResponse,
    CliMultiBytesResponse,
    CompressPdfRequest,
    ConvertDocumentRequest,
    ConvertPdfARequest,
    ConvertImageRequest,
    DownloadMediaRequest,
    ExtractPdfTextRequest,
    GenerateMediaThumbnailRequest,
    GenerateQrCodeRequest,
    GetMediaInfoRequest,
    GetMediaInfoResponse,
    MergePdfsRequest,
    OcrImageRequest,
    OptimizeImageRequest,
    ReadMediaMetadataRequest,
    ReadMediaMetadataResponse,
    RenderDiagramRequest,
    RenderPdfPagesRequest,
    RenderSvgRequest,
    TranscodeMediaRequest,
    ExtractAudioRequest,
    TrimMediaRequest,
    WriteMediaMetadataRequest,
)
from cli_service_pb2_grpc import CliServiceServicer

from cli_worker.service.sandbox import run_sandboxed
from cli_worker.service.command_builders.exiftool_builder import ExifToolCommandBuilder
from cli_worker.service.command_builders.ffmpeg_builder import FFmpegCommandBuilder
from cli_worker.service.command_builders.ghostscript_builder import GhostscriptCommandBuilder
from cli_worker.service.command_builders.graphviz_builder import GraphvizCommandBuilder
from cli_worker.service.command_builders.image_optimizer_builder import ImageOptimizerCommandBuilder
from cli_worker.service.command_builders.imagemagick_builder import ImageMagickCommandBuilder
from cli_worker.service.command_builders.pandoc_builder import PandocCommandBuilder
from cli_worker.service.command_builders.poppler_builder import PopplerCommandBuilder
from cli_worker.service.command_builders.qrencode_builder import QrencodeCommandBuilder
from cli_worker.service.command_builders.rsvg_builder import RsvgCommandBuilder
from cli_worker.service.command_builders.tesseract_builder import TesseractCommandBuilder
from cli_worker.service.command_builders.ytdlp_builder import YtDlpCommandBuilder

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = int(os.getenv("CLI_DEFAULT_TIMEOUT", "120"))
FFMPEG_TIMEOUT = 300
YTDLP_TIMEOUT = 600
MAX_OUTPUT_MB = int(os.getenv("CLI_MAX_OUTPUT_MB", "100"))
MAX_OUTPUT_BYTES = MAX_OUTPUT_MB * 1024 * 1024


def _scratch_dir() -> str:
    return os.getenv("CLI_SCRATCH_DIR", "/scratch")


def _job_dir() -> str:
    base = _scratch_dir()
    job_id = uuid.uuid4().hex
    path = os.path.join(base, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_cleanup(path: str) -> None:
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        logger.warning("Cleanup failed for %s: %s", path, e)


def _bytes_response(success: bool, output_data: bytes = b"", output_filename: str = "", error: str = "", formatted: str = "") -> CliBytesResponse:
    return CliBytesResponse(
        success=success,
        output_data=output_data,
        output_filename=output_filename,
        error=error or None,
        formatted=formatted or None,
    )


class CliServiceImplementation(CliServiceServicer):
    async def TranscodeMedia(self, request: TranscodeMediaRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            out_path = os.path.join(job_dir, "out." + (request.output_format or "mp4").lower())
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = FFmpegCommandBuilder.build_transcode(
                in_path, out_path, request.output_format or "mp4",
                audio_bitrate_kbps=request.audio_bitrate_kbps or None,
                video_bitrate_kbps=request.video_bitrate_kbps or None,
                max_width=request.max_width or None,
                max_height=request.max_height or None,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=FFMPEG_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Transcode failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=os.path.basename(out_path), formatted=f"Transcoded to {request.output_format}.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def ExtractAudio(self, request: ExtractAudioRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            out_path = os.path.join(job_dir, "out." + (request.output_format or "mp3").lower())
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = FFmpegCommandBuilder.build_extract_audio(
                in_path, out_path, request.output_format or "mp3",
                bitrate_kbps=request.bitrate_kbps or None,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=FFMPEG_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Extract audio failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=os.path.basename(out_path), formatted=f"Extracted audio as {request.output_format}.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def GenerateMediaThumbnail(self, request: GenerateMediaThumbnailRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            out_path = os.path.join(job_dir, "thumb." + (request.output_format or "jpg").lower())
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = FFmpegCommandBuilder.build_thumbnail(
                in_path, out_path, request.output_format or "jpg",
                width=request.width or 320, height=request.height or 240,
                timestamp=request.timestamp if request.timestamp else None,
                quality=request.quality if request.quality else None,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=DEFAULT_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Thumbnail failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=os.path.basename(out_path), formatted="Generated thumbnail.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def GetMediaInfo(self, request: GetMediaInfoRequest, context: grpc.aio.ServicerContext) -> GetMediaInfoResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = FFmpegCommandBuilder.build_probe(in_path)
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=30, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return GetMediaInfoResponse(success=False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Probe failed.")
            try:
                info = json.loads(result.stdout.decode("utf-8"))
                duration = None
                width = None
                height = None
                video_codec = None
                audio_codec = None
                audio_bitrate_kbps = None
                video_bitrate_kbps = None
                for s in info.get("streams", []):
                    if s.get("codec_type") == "video":
                        if width is None:
                            width = int(s.get("width", 0) or 0)
                            height = int(s.get("height", 0) or 0)
                            video_codec = s.get("codec_name") or ""
                            video_bitrate_kbps = int((int(s.get("bit_rate", 0) or 0) / 1000))
                    elif s.get("codec_type") == "audio":
                        if audio_codec is None:
                            audio_codec = s.get("codec_name") or ""
                            audio_bitrate_kbps = int((int(s.get("bit_rate", 0) or 0) / 1000))
                fmt = info.get("format", {})
                duration = float(fmt.get("duration", 0) or 0)
                return GetMediaInfoResponse(
                    success=True,
                    formatted=json.dumps(info, indent=2)[:2000],
                    duration_seconds=duration,
                    width=width or None,
                    height=height or None,
                    video_codec=video_codec or None,
                    audio_codec=audio_codec or None,
                    audio_bitrate_kbps=audio_bitrate_kbps or None,
                    video_bitrate_kbps=video_bitrate_kbps or None,
                    metadata_json=result.stdout.decode("utf-8"),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                return GetMediaInfoResponse(success=False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def TrimMedia(self, request: TrimMediaRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            out_path = os.path.join(job_dir, "out." + (request.output_format or "mp4").lower())
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = FFmpegCommandBuilder.build_trim(
                in_path, out_path, request.output_format or "mp4",
                request.start_time or "0", request.end_time or "0",
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=FFMPEG_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Trim failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=os.path.basename(out_path), formatted="Trimmed media.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def BurnSubtitles(self, request: BurnSubtitlesRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input.mp4")
            sub_path = os.path.join(job_dir, request.subtitle_filename or "subs.srt")
            out_path = os.path.join(job_dir, "out." + (request.output_format or "mp4").lower())
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            with open(sub_path, "wb") as f:
                f.write(request.subtitle_data)
            font_size = request.font_size if request.HasField("font_size") and request.font_size else None
            font_color = request.font_color if request.HasField("font_color") and request.font_color else None
            cmd = FFmpegCommandBuilder.build_burn_subtitles(
                in_path, sub_path, out_path, request.output_format or "mp4",
                font_size=font_size, font_color=font_color,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=FFMPEG_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Burn subtitles failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=os.path.basename(out_path), formatted="Burned subtitles into video.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def ConvertImage(self, request: ConvertImageRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            ext = (request.output_format or "png").lower()
            if ext == "jpeg":
                ext = "jpg"
            out_path = os.path.join(job_dir, "out." + ext)
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = ImageMagickCommandBuilder.build_convert(
                in_path, out_path, request.output_format or "png",
                width=request.width or None, height=request.height or None,
                quality=request.quality or None,
                crop_x=request.crop_x or None, crop_y=request.crop_y or None,
                crop_width=request.crop_width or None, crop_height=request.crop_height or None,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=DEFAULT_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Convert image failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=os.path.basename(out_path), formatted=f"Converted to {request.output_format}.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def OptimizeImage(self, request: OptimizeImageRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            fmt = ImageOptimizerCommandBuilder.format_from_filename(request.input_filename or "input")
            strip = bool(request.strip_metadata)
            if fmt == "png":
                cmd = ImageOptimizerCommandBuilder.build_optipng(in_path, strip_metadata=strip)
            else:
                cmd = ImageOptimizerCommandBuilder.build_jpegoptim(
                    in_path,
                    quality=request.quality if request.quality else None,
                    strip_metadata=strip,
                )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=60, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Optimize image failed.")
            with open(in_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=os.path.basename(in_path), formatted=f"Optimized {fmt} image.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def ConvertDocument(self, request: ConvertDocumentRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            out_name = request.output_filename or ("out." + (request.output_format or "pdf").lower())
            out_path = os.path.join(job_dir, out_name)
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = PandocCommandBuilder.build_convert(
                in_path, out_path,
                request.input_format or "markdown",
                request.output_format or "pdf",
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=DEFAULT_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Document conversion failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=os.path.basename(out_path), formatted=f"Converted to {request.output_format}.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def OcrImage(self, request: OcrImageRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            out_base = os.path.join(job_dir, "out")
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = TesseractCommandBuilder.build_ocr(
                in_path, out_base, request.output_format or "text",
                languages=list(request.languages) if request.languages else None,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=120, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "OCR failed.")
            ext = {"text": ".txt", "hocr": ".hocr", "tsv": ".tsv", "pdf": ".pdf"}.get(request.output_format or "text", ".txt")
            out_path = out_base + ext
            if not os.path.isfile(out_path):
                return _bytes_response(False, error="OCR output file not found", formatted="OCR output file not found.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename="out" + ext, formatted="OCR completed.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def ExtractPdfText(self, request: ExtractPdfTextRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input.pdf")
            out_path = os.path.join(job_dir, "out.txt")
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = PopplerCommandBuilder.build_pdftotext(
                in_path, out_path,
                first_page=request.first_page or None,
                last_page=request.last_page or None,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=DEFAULT_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Extract PDF text failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename="out.txt", formatted="Extracted PDF text.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def SplitPdf(self, request: SplitPdfRequest, context: grpc.aio.ServicerContext) -> CliMultiBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input.pdf")
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            pattern = os.path.join(job_dir, "page_%04d.pdf")
            cmd = PopplerCommandBuilder.build_pdfseparate(in_path, pattern)
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=DEFAULT_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return CliMultiBytesResponse(success=False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Split PDF failed.")
            outputs = []
            names = []
            i = 1
            while True:
                p = os.path.join(job_dir, f"page_{i:04d}.pdf")
                if not os.path.isfile(p):
                    break
                with open(p, "rb") as f:
                    outputs.append(f.read())
                names.append(os.path.basename(p))
                i += 1
            if not outputs:
                return CliMultiBytesResponse(success=False, error="No pages produced", formatted="No pages produced.")
            return CliMultiBytesResponse(success=True, output_data=outputs, output_filenames=names, formatted=f"Split into {len(outputs)} page(s).")
        except ValueError as e:
            return CliMultiBytesResponse(success=False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def MergePdfs(self, request: MergePdfsRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            if not request.input_documents or len(request.input_documents) < 2:
                return _bytes_response(False, error="At least 2 PDFs required to merge", formatted="At least 2 PDFs required to merge.")
            paths = []
            for i, data in enumerate(request.input_documents):
                p = os.path.join(job_dir, f"in_{i}.pdf")
                with open(p, "wb") as f:
                    f.write(data)
                paths.append(p)
            out_path = os.path.join(job_dir, "merged.pdf")
            cmd = PopplerCommandBuilder.build_pdfunite(paths, out_path)
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=DEFAULT_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Merge PDFs failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename="merged.pdf", formatted=f"Merged {len(paths)} PDF(s).")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def RenderPdfPages(self, request: RenderPdfPagesRequest, context: grpc.aio.ServicerContext) -> CliMultiBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input.pdf")
            out_prefix = os.path.join(job_dir, "page")
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            fmt = (request.output_format or "png").lower()
            if fmt == "jpeg":
                fmt = "jpeg"
            else:
                fmt = "png"
            cmd = PopplerCommandBuilder.build_pdftoppm(
                in_path, out_prefix, fmt,
                first_page=request.first_page or None,
                last_page=request.last_page or None,
                dpi=request.dpi or 150,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=DEFAULT_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return CliMultiBytesResponse(success=False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Render PDF pages failed.")
            ext = ".png" if fmt == "png" else ".jpg"
            outputs = []
            names = []
            pattern = os.path.join(job_dir, "page-*" + ext)
            for p in sorted(glob.glob(pattern)):
                with open(p, "rb") as f:
                    outputs.append(f.read())
                names.append(os.path.basename(p))
            if not outputs:
                return CliMultiBytesResponse(success=False, error="No pages produced", formatted="No pages produced.")
            return CliMultiBytesResponse(success=True, output_data=outputs, output_filenames=names, formatted=f"Rendered {len(outputs)} page(s) to {fmt}.")
        except ValueError as e:
            return CliMultiBytesResponse(success=False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def RenderDiagram(self, request: RenderDiagramRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, "input.dot")
            ext = (request.output_format or "png").lower()
            out_path = os.path.join(job_dir, "out." + ext)
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = GraphvizCommandBuilder.build_render(
                in_path, out_path,
                request.output_format or "png",
                request.engine or "dot",
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=30, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Render diagram failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename="out." + ext, formatted=f"Rendered diagram as {request.output_format}.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def ReadMediaMetadata(self, request: ReadMediaMetadataRequest, context: grpc.aio.ServicerContext) -> ReadMediaMetadataResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = ExifToolCommandBuilder.build_read(in_path)
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=10, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return ReadMediaMetadataResponse(success=False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Read metadata failed.")
            return ReadMediaMetadataResponse(
                success=True,
                metadata_json=result.stdout.decode("utf-8"),
                formatted=result.stdout.decode("utf-8", errors="replace")[:1000],
            )
        except ValueError as e:
            return ReadMediaMetadataResponse(success=False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def WriteMediaMetadata(self, request: WriteMediaMetadataRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input")
            out_path = os.path.join(job_dir, "out." + (os.path.splitext(request.input_filename or "input")[1] or ".jpg"))
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            meta = dict(request.metadata_fields or {})
            if not meta:
                return _bytes_response(False, error="No metadata fields provided", formatted="No metadata fields provided.")
            cmd = ExifToolCommandBuilder.build_write(in_path, out_path, meta)
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=10, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Write metadata failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=os.path.basename(out_path), formatted="Metadata written.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def DownloadMedia(self, request: DownloadMediaRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            out_tpl = os.path.join(job_dir, "out.%(ext)s")
            max_mb = request.max_filesize_mb if request.max_filesize_mb else int(os.getenv("YTDLP_MAX_FILESIZE_MB", "500"))
            cmd = YtDlpCommandBuilder.build_download(
                request.url,
                out_tpl,
                format_preference=request.format_preference or "best",
                max_filesize_mb=max_mb,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=YTDLP_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Download failed.")
            out_files = [f for f in os.listdir(job_dir) if f.startswith("out.")]
            if not out_files:
                return _bytes_response(False, error="No output file from yt-dlp", formatted="No output file from yt-dlp.")
            out_path = os.path.join(job_dir, out_files[0])
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=out_files[0], formatted=f"Downloaded: {out_files[0]}.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def CompressPdf(self, request: CompressPdfRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input.pdf")
            out_path = os.path.join(job_dir, "compressed.pdf")
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = GhostscriptCommandBuilder.build_compress(
                in_path, out_path, request.quality or "ebook",
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=DEFAULT_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Compress PDF failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename="compressed.pdf", formatted=f"Compressed PDF ({request.quality or 'ebook'}).")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def ConvertPdfA(self, request: ConvertPdfARequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, request.input_filename or "input.pdf")
            out_path = os.path.join(job_dir, "output.pdf")
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = GhostscriptCommandBuilder.build_pdfa(in_path, out_path)
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=DEFAULT_TIMEOUT, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Convert PDF/A failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename="output.pdf", formatted="Converted to PDF/A-2b.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def GenerateQrCode(self, request: GenerateQrCodeRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            ext = (request.output_format or "png").lower()
            if ext == "svg":
                out_name = "qrcode.svg"
            else:
                out_name = "qrcode.png"
            out_path = os.path.join(job_dir, out_name)
            cmd = QrencodeCommandBuilder.build_qrencode(
                request.content,
                out_path,
                request.output_format or "png",
                size=request.size or 256,
                error_correction=request.error_correction or "M",
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=30, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Generate QR code failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename=out_name, formatted="Generated QR code.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)

    async def RenderSvg(self, request: RenderSvgRequest, context: grpc.aio.ServicerContext) -> CliBytesResponse:
        job_dir = _job_dir()
        try:
            in_path = os.path.join(job_dir, "input.svg")
            ext = (request.output_format or "png").lower()
            out_path = os.path.join(job_dir, "out." + ext)
            with open(in_path, "wb") as f:
                f.write(request.input_data)
            cmd = RsvgCommandBuilder.build_rsvg_convert(
                in_path, out_path,
                request.output_format or "png",
                width=request.width or None,
                height=request.height or None,
                dpi=request.dpi or None,
            )
            result = await run_sandboxed(cmd, job_dir, timeout_seconds=30, max_stdout_bytes=MAX_OUTPUT_BYTES)
            if not result.success:
                return _bytes_response(False, error=result.error_message or result.stderr.decode("utf-8", errors="replace"), formatted=result.error_message or "Render SVG failed.")
            with open(out_path, "rb") as f:
                data = f.read()
            return _bytes_response(True, output_data=data, output_filename="out." + ext, formatted=f"Rendered SVG to {request.output_format}.")
        except ValueError as e:
            return _bytes_response(False, error=str(e), formatted=str(e))
        finally:
            _safe_cleanup(job_dir)


async def serve(port: int) -> None:
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", MAX_OUTPUT_BYTES),
            ("grpc.max_receive_message_length", MAX_OUTPUT_BYTES),
        ]
    )
    from cli_service_pb2_grpc import add_CliServiceServicer_to_server
    add_CliServiceServicer_to_server(CliServiceImplementation(), server)
    listen_addr = f"0.0.0.0:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("CLI Worker gRPC server listening on %s", listen_addr)
    await server.wait_for_termination()
