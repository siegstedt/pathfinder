import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from services.api.file_load import FileLoader
from services.entity_resolution import EntityResolution
from services.extraction import NodeExtractor

file_loader = FileLoader()
node_extractor = NodeExtractor()
entity_resolution = EntityResolution()

if __name__ == "__main__":
    # pdf = file_loader.get_file()
    print("file loaded")
    # node_extractor.extract(pdf)
    print("finished extract")
    entity_resolution.run_entity_resolution()
