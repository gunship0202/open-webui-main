import os
import shutil
from pathlib import Path
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pydantic import BaseModel, Field
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

class Tools:
    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    class Valves(BaseModel):
        output_dir: str = Field("./exports", description="文件导出的默认目录路径")
        default_font: str = Field("Helvetica", description="PDF导出时使用的默认字体")

    class UserValves(BaseModel):
        pdf_font_size: int = Field(12, description="PDF文件的字体大小")
        line_spacing: int = Field(20, description="PDF文件的行间距")

    def _ensure_output_dir(self) -> Path:
        output_dir = Path(self.valves.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_as_markdown(self, text: str, filename: str = "output.md") -> str:
        try:
            output_path = self._ensure_output_dir() / filename
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            return str(output_path)
        except Exception as e:
            raise Exception(f"保存Markdown文件时发生错误: {str(e)}")

    def save_as_word(self, text: str, filename: str = "output.docx") -> str:
        try:
            doc = Document()
            for paragraph in text.split("\n"):
                if paragraph.strip():
                    doc.add_paragraph(paragraph)

            output_path = self._ensure_output_dir() / filename
            doc.save(output_path)
            return str(output_path)
        except Exception as e:
            raise Exception(f"保存Word文件时发生错误: {str(e)}")

    def save_as_pdf(self, text: str, filename: str = "output.pdf") -> str:
        try:
            output_path = self._ensure_output_dir() / filename
            pdf = canvas.Canvas(str(output_path), pagesize=A4)

            pdf.setFont(self.valves.default_font, self.user_valves.pdf_font_size)

            lines = text.split("\n")
            y_position = A4[1] - 50

            for line in lines:
                if y_position < 50:
                    pdf.showPage()
                    pdf.setFont(
                        self.valves.default_font, self.user_valves.pdf_font_size
                    )
                    y_position = A4[1] - 50

                pdf.drawString(50, y_position, line)
                y_position -= self.user_valves.line_spacing

            pdf.save()
            return str(output_path)
        except Exception as e:
            raise Exception(f"保存PDF文件时发生错误: {str(e)}")

    def start_download_server(self, directory: str = "./exports", port: int = 8000):
        """
        启动简单的 HTTP 服务器，提供下载文件的链接
        """
        os.chdir(directory)  # 设置工作目录为文件导出的目录
        handler = SimpleHTTPRequestHandler
        with TCPServer(("", port), handler) as httpd:
            print(f"HTTP 服务器已启动，提供下载服务：http://localhost:{port}/")
            httpd.serve_forever()


if __name__ == "__main__":
    tools = Tools()
    text_to_export = "这是一些示例文本，您可以将其导出为不同的格式。"

    # 导出为 Markdown
    markdown_path = tools.save_as_markdown(text_to_export, "output.md")
    print(f"Markdown 文件已保存到: {markdown_path}")

    # 导出为 Word
    word_path = tools.save_as_word(text_to_export, "output.docx")
    print(f"Word 文件已保存到: {word_path}")

    # 导出为 PDF
    pdf_path = tools.save_as_pdf(text_to_export, "output.pdf")
    print(f"PDF 文件已保存到: {pdf_path}")

    # 启动下载服务器
    tools.start_download_server()

    print(f"您可以通过以下链接下载文件：")
    print(f"Markdown: http://localhost:8000/{Path(markdown_path).name}")
    print(f"Word: http://localhost:8000/{Path(word_path).name}")
    print(f"PDF: http://localhost:8000/{Path(pdf_path).name}") 