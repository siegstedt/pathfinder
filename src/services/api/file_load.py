import fitz


class FileLoader:
    def __init__(self) -> None:
        self.local_path = "src/data/lastenheft.pdf"

    def get_file(self):
        return self._load_local()

    def _load_local(self, pdf_path=False):
        if not pdf_path:
            pdf_path = self.local_path
        pdf_document = fitz.open(pdf_path)
        return pdf_document
