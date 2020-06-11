
from tqhub.ws.dataset.servies import ProcessFlowService
from tqhub.ws.dataset.types import Flow, WGSCoordinates
from process import process as p

class ProcessFlowServiceImpl(ProcessFlowService):
    def __init__(self):

        print("# ---------------------------------------------------------------------------- #")
        print("#                            ProcessFlowServiceImpl                            #")
        print("# ---------------------------------------------------------------------------- #")
    
    def process(self, *, para: Flow) -> None:
        p(
            Flow.vector_data_source,
            WgsCord=Flow.cord,
            Class_key=Flow.class_key,
            DataSetName=Flow.data_set_name,
            Merge=Flow.merge,
            Keep_local=Flow.keeplocal,
            remote_dataset_root=Flow.remote_dataset_local,
            over_write=Flow.overwrite
        )
