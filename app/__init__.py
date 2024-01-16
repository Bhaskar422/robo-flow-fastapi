
from dotenv import load_dotenv
from roboflow import Roboflow
import os

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

# Define the project and model
project_name = os.getenv("ROBOFLOW_PROJECT_NAME")
version_number = os.getenv("ROBOFLOW_MODEL_VERSION_NUMBER")

project = rf.workspace().project(project_name)
model = project.version(version_number).model