import fitz


class FileLoader:
    def __init__(self) -> None:
        self.local_path = "src/data/lastenheft.pdf"

    def get_file(self, path=False):
        return self._load_file(path)

    def _load_file(self, pdf_path):
        if not pdf_path:
            pdf_path = self.local_path
        pdf_document = fitz.open(pdf_path)
        return pdf_document
