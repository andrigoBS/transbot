from PIL import Image, ImageFont, ImageDraw

from src.file_path_helper import FilePathHelper


class Summary:
    def __init__(self):
        self.summary = None

    def set_summary(self, summary):
        self.summary = summary
        return self

    def get_text(self):
        return "\n".join(self.summary)

    def plot_img(self):
        summary_string = self.get_text()

        try:
            try:
                font = ImageFont.truetype('consola.ttf', 18)
            except:
                font = ImageFont.load_default()

            line_width, line_height = font.getsize(max(self.summary))
        except:
            return self

        img_height = line_height * len(self.summary)
        img = Image.new('RGBA', (line_width + 8, img_height + 8), "#FFF")
        draw = ImageDraw.Draw(img)
        draw.text((4, 4), summary_string, "#000", font=font)
        img.save(FilePathHelper().get_reports_file_path('summary'))
        return self
