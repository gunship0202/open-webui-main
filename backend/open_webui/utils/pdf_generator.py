from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List
from html import escape

from markdown import markdown

import site
from fpdf import FPDF

from open_webui.env import STATIC_DIR, FONTS_DIR
from open_webui.models.chats import ChatTitleMessagesForm


class PDFGenerator:
    """
    Description:
    The `PDFGenerator` class is designed to create PDF documents from chat messages.
    The process involves transforming markdown content into HTML and then into a PDF format

    Attributes:
    - `form_data`: An instance of `ChatTitleMessagesForm` containing title and messages.

    """

    def __init__(self, form_data: ChatTitleMessagesForm):
        self.html_body = None
        self.messages_html = None
        self.form_data = form_data

        self.css = Path(STATIC_DIR / "assets" / "pdf-style.css").read_text()

    def format_timestamp(self, timestamp: float) -> str:
        """Convert a UNIX timestamp to a formatted date string."""
        try:
            date_time = datetime.fromtimestamp(timestamp)
            return date_time.strftime("%Y-%m-%d, %H:%M:%S")
        except (ValueError, TypeError) as e:
            # Log the error if necessary
            return ""

    def _build_html_message(self, message: Dict[str, Any]) -> str:
        """Build HTML for a single message."""
        role = escape(message.get("role", "user"))
        content = escape(message.get("content", ""))
        timestamp = message.get("timestamp")

        model = escape(message.get("model") if role == "assistant" else "")

        date_str = escape(self.format_timestamp(timestamp) if timestamp else "")

        # extends pymdownx extension to convert markdown to html.
        # - https://facelessuser.github.io/pymdown-extensions/usage_notes/
        # html_content = markdown(content, extensions=["pymdownx.extra"])

        content = content.replace("\n", "<br/>")
        html_message = f"""
            <div>
                <div>
                    <h4>
                        <strong>{role.title()}</strong>
                        <span style="font-size: 12px;">{model}</span>
                    </h4>
                    <div> {date_str} </div>
                </div>
                <br/>
                <br/>

                <div>
                    {content}
                </div>
            </div>
            <br/>
          """
        return html_message

    def _generate_html_body(self) -> str:
        """Generate the full HTML body for the PDF."""
        escaped_title = escape(self.form_data.title)
        return f"""
        <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            </head>
            <body>
            <div>
                <div>
                    <h2>{escaped_title}</h2>
                    {self.messages_html}
                </div>
            </div>
            </body>
        </html>
        """

    def generate_chat_pdf(self) -> bytes:
        """
        Generate a PDF from chat messages.
        """
        try:
            print("=== PDF生成开始 ===")
            print(f"FONTS_DIR路径: {FONTS_DIR}")
            
            pdf = FPDF()
            pdf.add_page()
            
            print("正在检查字体文件...")
            font_path = Path(f"{FONTS_DIR}/NotoSans-Regular.ttf")
            print(f"字体文件路径: {font_path}")
            print(f"字体文件是否存在: {font_path.exists()}")
            
            try:
                print("尝试加载 NotoSans 字体...")
                pdf.add_font("NotoSans", "", f"{FONTS_DIR}/NotoSans-Regular.ttf")
                pdf.add_font("NotoSans", "B", f"{FONTS_DIR}/NotoSans-Bold.ttf")
                pdf.add_font("NotoSans", "I", f"{FONTS_DIR}/NotoSans-Italic.ttf")
                print("NotoSans 字体加载成功")
                pdf.set_font("NotoSans", size=12)
            except Exception as font_error:
                print(f"加载 NotoSans 字体失败: {str(font_error)}")
                print("尝试使用 Arial 字体...")
                pdf.set_font("Arial", size=12)
            
            print("设置PDF属性...")
            pdf.set_auto_page_break(auto=True, margin=15)
            
            print("开始处理消息内容...")
            print(f"消息数量: {len(self.form_data.messages)}")
            
            messages_html_list = []
            for i, msg in enumerate(self.form_data.messages):
                print(f"处理第 {i+1} 条消息...")
                try:
                    html_msg = self._build_html_message(msg)
                    messages_html_list.append(html_msg)
                except Exception as msg_error:
                    print(f"处理消息 {i+1} 失败: {str(msg_error)}")
                    raise
            
            self.messages_html = "<div>" + "".join(messages_html_list) + "</div>"
            
            print("生成完整HTML...")
            self.html_body = self._generate_html_body()
            
            print("写入PDF内容...")
            try:
                pdf.write_html(self.html_body)
            except Exception as write_error:
                print(f"写入HTML内容失败: {str(write_error)}")
                raise
            
            print("生成PDF字节流...")
            try:
                pdf_bytes = pdf.output()
                print("PDF生成完成")
                return bytes(pdf_bytes)
            except Exception as output_error:
                print(f"输出PDF失败: {str(output_error)}")
                raise
            
        except Exception as e:
            print("\n=== PDF生成失败 ===")
            print(f"错误类型: {type(e)}")
            print(f"错误信息: {str(e)}")
            import traceback
            print(f"错误堆栈:\n{traceback.format_exc()}")
            raise
