from roboflow import Roboflow
rf = Roboflow(api_key="Dbbmn5uZ3G2QVQx8zIKi")
project = rf.workspace("roboflow-100").project("road-signs-6ih4y")
version = project.version(2)
dataset = version.download("yolov5")