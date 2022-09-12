from catalyst.core.runner import IRunner
from catalyst.runners.runner import Runner
from catalyst.runners.supervised import ISupervisedRunner
from catalyst.runners.supervised import SupervisedRunner

class CustomSupervisedRunner(SupervisedRunner, ISupervisedRunner, Runner):
    def __init__(
        self
    ):
        SupervisedRunner.__init__(
            self,
            model: RunnerModel = None,
            engine: Engine = None,
            input_key: Any = "features",
            output_key: Any = "logits",
            target_key: str = "targets",
            loss_key: str = "loss",
        )
        ISupervisedRunner.__init__(
                self,
                input_key=input_key,
                output_key=output_key,
                target_key=target_key,
                loss_key=loss_key,
            )
            Runner.__init__(self, model=model, engine=engine)