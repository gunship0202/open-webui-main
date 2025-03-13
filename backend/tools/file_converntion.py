"""
title: MD File Conversion Action
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1
required_open_webui_version: 0.5.20
"""

from pydantic import BaseModel, Field
from typing import Optional, Tuple
import re
import os
from pathlib import Path
import markdown

class Action:
    class Valves(BaseModel):
        storage_path: str = Field(
            default="storage/markdown",
            description="Path to store generated markdown files"
        )
        base_url: str = Field(
            default="/download",
            description="Base URL for file downloads"
        )

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True,
            description="Show status of the action"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _ensure_storage_path(self) -> Path:
        storage_path = Path(self.valves.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    def create_things_md(self, markdown_list: str) -> str:
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

        # 保存 markdown 文件
        storage_path = self._ensure_storage_path()
        md_filename = f"things_{hash(markdown_list)}.md"
        md_path = storage_path / md_filename

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return f"{self.valves.base_url}/{md_filename}"

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
                                "description": "正在生成檔案",
                                "done": False,
                            },
                        }
                    )

                last_assistant_message = body["messages"][-1]
                md_url = self.create_things_md(last_assistant_message["content"])

                if user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "檔案生成完成",
                                "done": True,
                            },
                        }
                    )

                content = f"\n- [下載 Markdown 檔案]({md_url})"
                
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": content
                        },
                    }
                )

        except Exception as e:
            error_msg = f"檔案生成時發生錯誤: {str(e)}"
            print(error_msg)
            if __event_emitter__ and user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "檔案生成時發生錯誤",
                            "done": True,
                        },
                    }
                )
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": "Error:檔案生成"},
                            "document": [error_msg],
                            "metadata": [{"source": "檔案生成器"}],
                        },
                    }
                )

        return None