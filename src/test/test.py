import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from services.api.file_load import FileLoader
from services.entity_resolution import EntityResolution
from services.extraction import NodeExtractor
from services.community_construction import CommunityBuilder

file_loader = FileLoader()
node_extractor = NodeExtractor()
entity_resolution = EntityResolution()
community_builder = CommunityBuilder()

if __name__ == "__main__":
    """
    for file in [
        "/home/simon/Documents/Pure_Inference/Project Pathfinder/Backend/src/data/disclosure_02832977.pdf",
        # "/home/simon/Documents/Pure_Inference/Project Pathfinder/Backend/src/data/disclosure_02834651.pdf",
        # "/home/simon/Documents/Pure_Inference/Project Pathfinder/Backend/src/data/disclosure_02675210.pdf",
        # "/home/simon/Documents/Pure_Inference/Project Pathfinder/Backend/src/data/disclosure_02677626.pdf",
        # "/home/simon/Documents/Pure_Inference/Project Pathfinder/Backend/src/data/disclosure_02678346.pdf",
    ]:
        pdf_name = str(file).split("/")[-1]
        pdf_name = pdf_name.split(".")[0]
        pdf_name = pdf_name.split("_")[-1]
        pdf = file_loader.get_file(file)
        node_extractor.extract(pdf, pdf_name=pdf_name, pdf_name_key="disclosure_id")
        print("finished extract")
    """
    # entity_resolution.run_entity_resolution()
    # print("finished entity")
    # entity_resolution.edge_resolution()
    community_builder.make_communities()
