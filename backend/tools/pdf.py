"""
title: PDF File Conversion Action
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1
required_open_webui_version: 0.5.20
"""

from pydantic import BaseModel, Field
from typing import Optional
import re
from pathlib import Path
import markdown_pdf
import tempfile
import os
from open_webui.main import DOWNLOAD_FOLDER


class Action:
    class Valves(BaseModel):
        storage_path: str = Field(
            default=str(DOWNLOAD_FOLDER),
            description="Path to store generated PDF files"
        )
        base_url: str = Field(
            default="/download",
            description="Base URL for file downloads"
        )
        pdf_css: str = Field(
            default="""
            body { font-family: Arial, sans-serif; line-height: 1.6; }
            h1 { color: #2c3e50; border-bottom: 2px solid #eee; }
            h2 { color: #34495e; margin-top: 20px; }
            input[type="checkbox"] { margin-right: 10px; }
            """,
            description="CSS styles for PDF generation"
        )

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True,
            description="Show status of the action"
        )
        generate_pdf: bool = Field(
            default=True,
            description="Whether to generate PDF along with Markdown"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _ensure_storage_path(self) -> Path:
        storage_path = Path(self.valves.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    def _convert_to_pdf(self, markdown_content: str, output_path: Path) -> None:
        try:
            # 创建 MarkdownPdf 实例
            pdf = markdown_pdf.MarkdownPdf()
            # 添加内容作为一个章节
            pdf.add_section(markdown_pdf.Section(markdown_content, toc=False))
            # 设置 CSS 样式
            pdf.css = self.valves.pdf_css
            # 保存 PDF
            pdf.save(str(output_path))
        except Exception as e:
            import traceback
            print(f"PDF 生成失敗: {str(e)}")
            print(f"詳細錯誤: {traceback.format_exc()}")
            raise

    def create_things_md(self, markdown_list: str) -> Optional[str]:
        # 處理內容，移除不必要的部分
        lines = markdown_list.split('\n')
        content_start = False
        content_end = False
        filtered_lines = []
        
        for line in lines:
            # 跳過開頭的說明文字
            if line.strip() == '```markdown':
                content_start = True
                continue
            # 跳過結尾的下載連結
            elif line.strip() == '```' or line.startswith('- [下載'):
                content_end = True
                continue
            
            if content_start and not content_end:
                filtered_lines.append(line)
        
        md_content = '\n'.join(filtered_lines)

        # Save directly as PDF
        storage_path = self._ensure_storage_path()
        pdf_filename = f"things_{hash(markdown_list)}.pdf"
        pdf_path = storage_path / pdf_filename

        try:
            self._convert_to_pdf(md_content, pdf_path)
            return f"{self.valves.base_url}/{pdf_filename}"
        except Exception as e:
            print(f"PDF 生成失敗: {str(e)}")
            return None

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        try:
            user_valves = __user__.get("valves") if __user__ else self.UserValves()

            if __event_emitter__:
                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "正在生成 PDF 檔案",
                                "done": False,
                            },
                        }
                    )

                last_assistant_message = body["messages"][-1]
                download_url = self.create_things_md(last_assistant_message["content"])

                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "PDF 檔案生成完成",
                                "done": True,
                            },
                        }
                    )

                if download_url:
                    content = f"\n- [下載 PDF 檔案]({download_url})"
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": content
                            },
                        }
                    )

        except Exception as e:
            error_msg = f"PDF 檔案生成時發生錯誤: {str(e)}"
            print(error_msg)
            if __event_emitter__ and user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "PDF 檔案生成時發生錯誤",
                            "done": True,
                        },
                    }
                )
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": "Error:PDF生成"},
                            "document": [error_msg],
                            "metadata": [{"source": "PDF生成器"}],
                        },
                    }
                )

        return None
