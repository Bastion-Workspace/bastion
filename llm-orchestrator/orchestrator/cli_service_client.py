"""
CLI Service Client - gRPC client for the CLI worker (sandboxed CLI tools).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import grpc
from protos import cli_service_pb2, cli_service_pb2_grpc

logger = logging.getLogger(__name__)

# 100 MB to match tools-service and cli-worker limits
_GRPC_MESSAGE_LIMIT = 100 * 1024 * 1024

_cli_client: Optional["CliServiceClient"] = None


class CliServiceClient:
    """gRPC client for the CLI worker service."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self.host = host or os.getenv("CLI_WORKER_HOST", "cli-worker")
        self.port = port or int(os.getenv("CLI_WORKER_PORT", "50060"))
        self.address = f"{self.host}:{self.port}"
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[cli_service_pb2_grpc.CliServiceStub] = None
        logger.info("CLI Service Client configured for %s", self.address)

    async def connect(self) -> None:
        if self._channel is None:
            options = [
                ("grpc.max_send_message_length", _GRPC_MESSAGE_LIMIT),
                ("grpc.max_receive_message_length", _GRPC_MESSAGE_LIMIT),
            ]
            self._channel = grpc.aio.insecure_channel(self.address, options=options)
            self._stub = cli_service_pb2_grpc.CliServiceStub(self._channel)

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None

    async def _ensure_connected(self) -> None:
        if self._stub is None:
            await self.connect()

    def _bytes_response(self, r: Any) -> Dict[str, Any]:
        return {
            "success": r.success,
            "output_data": bytes(r.output_data) if r.output_data else b"",
            "output_filename": r.output_filename or "",
            "error": r.error or "",
            "formatted": r.formatted or "",
        }

    async def transcode_media(
        self,
        input_data: bytes,
        input_filename: str,
        output_format: str,
        audio_bitrate_kbps: Optional[int] = None,
        video_bitrate_kbps: Optional[int] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.TranscodeMediaRequest(
            input_data=input_data,
            input_filename=input_filename,
            output_format=output_format,
            audio_bitrate_kbps=audio_bitrate_kbps or 0,
            video_bitrate_kbps=video_bitrate_kbps or 0,
            max_width=max_width or 0,
            max_height=max_height or 0,
        )
        if audio_bitrate_kbps is None:
            req.ClearField("audio_bitrate_kbps")
        if video_bitrate_kbps is None:
            req.ClearField("video_bitrate_kbps")
        if max_width is None:
            req.ClearField("max_width")
        if max_height is None:
            req.ClearField("max_height")
        r = await self._stub.TranscodeMedia(req)
        return self._bytes_response(r)

    async def extract_audio(
        self,
        input_data: bytes,
        input_filename: str,
        output_format: str,
        bitrate_kbps: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.ExtractAudioRequest(
            input_data=input_data,
            input_filename=input_filename,
            output_format=output_format,
            bitrate_kbps=bitrate_kbps or 0,
        )
        if bitrate_kbps is None:
            req.ClearField("bitrate_kbps")
        r = await self._stub.ExtractAudio(req)
        return self._bytes_response(r)

    async def generate_media_thumbnail(
        self,
        input_data: bytes,
        input_filename: str,
        output_format: str,
        width: int,
        height: int,
        timestamp: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.GenerateMediaThumbnailRequest(
            input_data=input_data,
            input_filename=input_filename,
            output_format=output_format,
            width=width,
            height=height,
            timestamp=timestamp or "",
            quality=quality or 0,
        )
        if timestamp is None:
            req.ClearField("timestamp")
        if quality is None:
            req.ClearField("quality")
        r = await self._stub.GenerateMediaThumbnail(req)
        return self._bytes_response(r)

    async def get_media_info(self, input_data: bytes, input_filename: str) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.GetMediaInfoRequest(input_data=input_data, input_filename=input_filename)
        r = await self._stub.GetMediaInfo(req)
        return {
            "success": r.success,
            "error": r.error or "",
            "formatted": r.formatted or "",
            "duration_seconds": r.duration_seconds if r.HasField("duration_seconds") else None,
            "width": r.width if r.HasField("width") else None,
            "height": r.height if r.HasField("height") else None,
            "video_codec": r.video_codec or None,
            "audio_codec": r.audio_codec or None,
            "audio_bitrate_kbps": r.audio_bitrate_kbps if r.HasField("audio_bitrate_kbps") else None,
            "video_bitrate_kbps": r.video_bitrate_kbps if r.HasField("video_bitrate_kbps") else None,
            "metadata_json": r.metadata_json or None,
        }

    async def trim_media(
        self,
        input_data: bytes,
        input_filename: str,
        output_format: str,
        start_time: str,
        end_time: str,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.TrimMediaRequest(
            input_data=input_data,
            input_filename=input_filename,
            output_format=output_format,
            start_time=start_time,
            end_time=end_time,
        )
        r = await self._stub.TrimMedia(req)
        return self._bytes_response(r)

    async def burn_subtitles(
        self,
        input_data: bytes,
        input_filename: str,
        subtitle_data: bytes,
        subtitle_filename: str,
        output_format: str,
        font_size: Optional[int] = None,
        font_color: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.BurnSubtitlesRequest(
            input_data=input_data,
            input_filename=input_filename,
            subtitle_data=subtitle_data,
            subtitle_filename=subtitle_filename,
            output_format=output_format,
        )
        if font_size is not None:
            req.font_size = font_size
        if font_color is not None:
            req.font_color = font_color
        r = await self._stub.BurnSubtitles(req)
        return self._bytes_response(r)

    async def convert_image(
        self,
        input_data: bytes,
        input_filename: str,
        output_format: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        quality: Optional[int] = None,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_width: Optional[int] = None,
        crop_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.ConvertImageRequest(
            input_data=input_data,
            input_filename=input_filename,
            output_format=output_format,
        )
        if width is not None:
            req.width = width
        if height is not None:
            req.height = height
        if quality is not None:
            req.quality = quality
        if crop_x is not None:
            req.crop_x = crop_x
        if crop_y is not None:
            req.crop_y = crop_y
        if crop_width is not None:
            req.crop_width = crop_width
        if crop_height is not None:
            req.crop_height = crop_height
        r = await self._stub.ConvertImage(req)
        return self._bytes_response(r)

    async def optimize_image(
        self,
        input_data: bytes,
        input_filename: str,
        quality: Optional[int] = None,
        strip_metadata: bool = False,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.OptimizeImageRequest(
            input_data=input_data,
            input_filename=input_filename,
            quality=quality or 0,
            strip_metadata=strip_metadata,
        )
        if quality is None:
            req.ClearField("quality")
        r = await self._stub.OptimizeImage(req)
        return self._bytes_response(r)

    async def convert_document(
        self,
        input_data: bytes,
        input_filename: str,
        input_format: str,
        output_format: str,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.ConvertDocumentRequest(
            input_data=input_data,
            input_filename=input_filename,
            input_format=input_format,
            output_format=output_format,
            output_filename=output_filename or "",
        )
        if output_filename is None:
            req.ClearField("output_filename")
        r = await self._stub.ConvertDocument(req)
        return self._bytes_response(r)

    async def ocr_image(
        self,
        input_data: bytes,
        input_filename: str,
        output_format: str = "text",
        languages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.OcrImageRequest(
            input_data=input_data,
            input_filename=input_filename,
            output_format=output_format,
            languages=languages or [],
        )
        r = await self._stub.OcrImage(req)
        return self._bytes_response(r)

    async def extract_pdf_text(
        self,
        input_data: bytes,
        input_filename: str,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.ExtractPdfTextRequest(
            input_data=input_data,
            input_filename=input_filename,
            first_page=first_page or 0,
            last_page=last_page or 0,
        )
        if first_page is None:
            req.ClearField("first_page")
        if last_page is None:
            req.ClearField("last_page")
        r = await self._stub.ExtractPdfText(req)
        return self._bytes_response(r)

    async def split_pdf(self, input_data: bytes, input_filename: str) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.SplitPdfRequest(input_data=input_data, input_filename=input_filename)
        r = await self._stub.SplitPdf(req)
        return {
            "success": r.success,
            "output_data": list(r.output_data) if r.output_data else [],
            "output_filenames": list(r.output_filenames) if r.output_filenames else [],
            "error": r.error or "",
            "formatted": r.formatted or "",
        }

    async def merge_pdfs(self, input_documents: List[bytes], input_filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.MergePdfsRequest(
            input_documents=input_documents,
            input_filenames=input_filenames or [],
        )
        r = await self._stub.MergePdfs(req)
        return self._bytes_response(r)

    def _multi_bytes_response(self, r: Any) -> Dict[str, Any]:
        return {
            "success": r.success,
            "output_data": [bytes(b) for b in r.output_data] if r.output_data else [],
            "output_filenames": list(r.output_filenames) if r.output_filenames else [],
            "error": r.error or "",
            "formatted": r.formatted or "",
        }

    async def render_pdf_pages(
        self,
        input_data: bytes,
        input_filename: str,
        output_format: str = "png",
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
        dpi: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.RenderPdfPagesRequest(
            input_data=input_data,
            input_filename=input_filename,
            output_format=output_format,
            first_page=first_page or 0,
            last_page=last_page or 0,
            dpi=dpi or 150,
        )
        if first_page is None:
            req.ClearField("first_page")
        if last_page is None:
            req.ClearField("last_page")
        if dpi is None:
            req.ClearField("dpi")
        r = await self._stub.RenderPdfPages(req)
        return self._multi_bytes_response(r)

    async def render_diagram(
        self,
        input_data: bytes,
        output_format: str,
        engine: str = "dot",
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.RenderDiagramRequest(
            input_data=input_data,
            output_format=output_format,
            engine=engine,
        )
        r = await self._stub.RenderDiagram(req)
        return self._bytes_response(r)

    async def read_media_metadata(self, input_data: bytes, input_filename: str) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.ReadMediaMetadataRequest(input_data=input_data, input_filename=input_filename)
        r = await self._stub.ReadMediaMetadata(req)
        return {
            "success": r.success,
            "error": r.error or "",
            "formatted": r.formatted or "",
            "metadata_json": r.metadata_json or "",
        }

    async def write_media_metadata(
        self,
        input_data: bytes,
        input_filename: str,
        metadata_fields: Dict[str, str],
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.WriteMediaMetadataRequest(
            input_data=input_data,
            input_filename=input_filename,
            metadata_fields=metadata_fields,
        )
        r = await self._stub.WriteMediaMetadata(req)
        return self._bytes_response(r)

    async def download_media(
        self,
        url: str,
        format_preference: str = "best",
        max_filesize_mb: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.DownloadMediaRequest(
            url=url,
            format_preference=format_preference,
            max_filesize_mb=max_filesize_mb or 0,
        )
        if max_filesize_mb is None:
            req.ClearField("max_filesize_mb")
        r = await self._stub.DownloadMedia(req)
        return self._bytes_response(r)

    async def compress_pdf(
        self,
        input_data: bytes,
        input_filename: str,
        quality: str = "ebook",
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.CompressPdfRequest(
            input_data=input_data,
            input_filename=input_filename,
            quality=quality,
        )
        r = await self._stub.CompressPdf(req)
        return self._bytes_response(r)

    async def convert_pdfa(self, input_data: bytes, input_filename: str) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.ConvertPdfARequest(
            input_data=input_data,
            input_filename=input_filename,
        )
        r = await self._stub.ConvertPdfA(req)
        return self._bytes_response(r)

    async def generate_qr_code(
        self,
        content: str,
        output_format: str = "png",
        size: Optional[int] = None,
        error_correction: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.GenerateQrCodeRequest(
            content=content,
            output_format=output_format,
            size=size or 256,
            error_correction=error_correction or "M",
        )
        if size is None:
            req.ClearField("size")
        if error_correction is None:
            req.ClearField("error_correction")
        r = await self._stub.GenerateQrCode(req)
        return self._bytes_response(r)

    async def render_svg(
        self,
        input_data: bytes,
        output_format: str = "png",
        width: Optional[int] = None,
        height: Optional[int] = None,
        dpi: Optional[int] = None,
    ) -> Dict[str, Any]:
        await self._ensure_connected()
        req = cli_service_pb2.RenderSvgRequest(
            input_data=input_data,
            output_format=output_format,
            width=width or 0,
            height=height or 0,
            dpi=dpi or 0,
        )
        if width is None:
            req.ClearField("width")
        if height is None:
            req.ClearField("height")
        if dpi is None:
            req.ClearField("dpi")
        r = await self._stub.RenderSvg(req)
        return self._bytes_response(r)


def get_cli_service_client() -> CliServiceClient:
    """Return the singleton CLI service client."""
    global _cli_client
    if _cli_client is None:
        _cli_client = CliServiceClient()
    return _cli_client
