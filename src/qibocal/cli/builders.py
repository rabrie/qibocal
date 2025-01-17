import datetime
import importlib
import inspect
import os
import shutil

import yaml

from qibocal import calibrations
from qibocal.config import log, raise_error
from qibocal.data import Data


def load_yaml(path):
    """Load yaml file from disk."""
    with open(path) as file:
        data = yaml.safe_load(file)
    return data


class ActionParser:
    """Class for parsing and executing single actions in the runcard."""

    def __init__(self, runcard, folder, name):
        self.runcard = runcard
        self.folder = folder
        self.func = None
        self.params = None
        self.name = name
        self.path = os.path.join(self.folder, f"data/{self.name}/")

        # FIXME: dummy fix
        self.__name__ = name

    def build(self):
        """Load function from :func:`qibocal.characterization.calibrations` and check arguments"""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # collect function from module
        self.func = getattr(calibrations, self.name)

        sig = inspect.signature(self.func)
        self.params = self.runcard["actions"][self.name]
        for param in list(sig.parameters)[2:-1]:
            if param not in self.params:
                raise_error(AttributeError, f"Missing parameter {param} in runcard.")

    def execute(self, data_format, platform, qubits):
        """Execute action and retrieve results."""
        if data_format is None:
            raise_error(ValueError, f"Cannot store data using {data_format} format.")

        results = self.func(platform, qubits, **self.params)

        for data in results:
            getattr(data, f"to_{data_format}")(self.path)


class niGSCactionParser(ActionParser):
    """ni = non interactive
    GSC = gate set characterization
    """

    def __init__(self, runcard, folder, name):
        super().__init__(runcard, folder, name)

        self.plots = []

        self.nqubits = self.runcard["actions"][self.name]["nqubits"]
        self.depths = self.runcard["actions"][self.name]["depths"]
        self.runs = self.runcard["actions"][self.name]["runs"]
        self.nshots = self.runcard["actions"][self.name]["nshots"]

        from qibocal.calibrations.niGSC.basics import noisemodels

        try:
            self.noise_params = self.runcard["actions"][self.name]["noise_params"]
        except KeyError:
            self.noise_params = None
        try:
            self.noise_model = getattr(
                noisemodels, self.runcard["actions"][self.name]["noise_model"]
            )(*self.noise_params)
        except:
            self.noise_model = None

    def load_plot(self):
        """Helper method to import the plotting function."""
        from qibocal.calibrations.niGSC.basics.plot import plot_qq

        self.plots.append((f"{self.name} protocol", plot_qq))

    def build(self):
        """Load appropirate module to run the experiment."""
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.module = importlib.import_module(f"qibocal.calibrations.niGSC.{self.name}")

    def execute(self, data_format, platform):
        """Executes a non-interactive gate set characterication using only the wanted
        module name.

        1. Build the circuit factory.
        2. Build the experiment object with the circuit factory.
        3. Execute the circuits generated by the circuit factory and store the wanted results.
        4. Post process the data.
        5. Store the experiment, if needed the circuit factory, and the aggregated data.

        Args:
            data_format (_type_): _description_
            platform (_type_): _description_
        """

        # Initiate the factory and the experiment.
        factory = self.module.ModuleFactory(
            self.nqubits, self.depths * self.runs, qubits=self.runcard["qubits"]
        )
        experiment = self.module.ModuleExperiment(
            factory, nshots=self.nshots, noise_model=self.noise_model
        )
        # Execute the circuits in the experiment.
        experiment.perform(experiment.execute)
        # Run the row by row postprocessing.
        self.module.post_processing_sequential(experiment)
        # Run aggregational tasks along with fitting.
        # This will return a data frame, store it right away.
        self.module.get_aggregational_data(experiment).to_pickle(
            f"{self.path}/fit_plot.pkl"
        )
        # Store the experiment.
        experiment.save(self.path)


class ActionBuilder:
    """Class for parsing and executing runcards.
    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already.
    """

    def __init__(self, runcard, folder=None, force=False):
        path, self.folder = self._generate_output_folder(folder, force)
        self.runcard = load_yaml(runcard)
        # Qibolab default backend if not provided in runcard.
        backend_name = self.runcard.get("backend", "qibolab")
        platform_name = self.runcard.get("platform", "dummy")
        platform_runcard = self.runcard.get("runcard", None)
        self.backend, self.platform = self._allocate_backend(
            backend_name, platform_name, path, platform_runcard
        )
        if self.platform is not None:
            self.qubits = {
                q: self.platform.qubits[q]
                for q in self.runcard["qubits"]
                if q in self.platform.qubits
            }
        else:
            self.qubits = self.runcard.get("qubits")
        self.format = self.runcard["format"]

        # Saving runcard
        shutil.copy(runcard, f"{path}/runcard.yml")
        self.save_meta(path, self.folder)

    @staticmethod
    def _generate_output_folder(folder, force):
        """Static method for generating the output folder.
        Args:
            folder (path): path for the output folder. If None it will be created a folder automatically
            force (bool): option to overwrite the output folder if it exists already.
        """
        if folder is None:
            import getpass

            e = datetime.datetime.now()
            user = getpass.getuser().replace(".", "-")
            date = e.strftime("%Y-%m-%d")
            folder = f"{date}-{'000'}-{user}"
            num = 0
            while os.path.exists(folder):
                log.info(f"Directory {folder} already exists.")
                num += 1
                folder = f"{date}-{str(num).rjust(3, '0')}-{user}"
                log.info(f"Trying to create directory {folder}")
        elif os.path.exists(folder) and not force:
            raise_error(RuntimeError, f"Directory {folder} already exists.")
        elif os.path.exists(folder) and force:
            log.warning(f"Deleting previous directory {folder}.")
            shutil.rmtree(os.path.join(os.getcwd(), folder))

        path = os.path.join(os.getcwd(), folder)
        log.info(f"Creating directory {folder}.")
        os.makedirs(path)
        return path, folder

    def _allocate_backend(self, backend_name, platform_name, path, platform_runcard):
        """Allocate the platform using Qibolab."""
        from qibo.backends import GlobalBackend, set_backend

        if backend_name == "qibolab":
            if platform_runcard is None:
                from qibolab.paths import qibolab_folder

                original_runcard = qibolab_folder / "runcards" / f"{platform_name}.yml"
            else:
                original_runcard = platform_runcard
            # copy of the original runcard that will stay unmodified
            shutil.copy(original_runcard, f"{path}/platform.yml")
            # copy of the original runcard that will be modified during calibration
            updated_runcard = f"{self.folder}/new_platform.yml"
            shutil.copy(original_runcard, updated_runcard)
            # allocate backend with updated_runcard
            set_backend(
                backend=backend_name, platform=platform_name, runcard=updated_runcard
            )
            backend = GlobalBackend()
            return backend, backend.platform
        else:
            set_backend(backend=backend_name, platform=platform_name)
            backend = GlobalBackend()
            return backend, None

    def save_meta(self, path, folder):
        import qibocal

        e = datetime.datetime.now(datetime.timezone.utc)
        meta = {}
        meta["title"] = folder
        meta["backend"] = str(self.backend)
        meta["platform"] = str(self.backend.platform)
        meta["date"] = e.strftime("%Y-%m-%d")
        meta["start-time"] = e.strftime("%H:%M:%S")
        meta["end-time"] = e.strftime("%H:%M:%S")
        meta["versions"] = self.backend.versions  # pylint: disable=E1101
        meta["versions"]["qibocal"] = qibocal.__version__

        with open(f"{path}/meta.yml", "w") as file:
            yaml.dump(meta, file)

    def execute(self):
        """Method to execute sequentially all the actions in the runcard."""
        if self.platform is not None:
            self.platform.connect()
            self.platform.setup()
            self.platform.start()

        actions = []
        for action in self.runcard["actions"]:
            actions.append(action)
            try:
                parser = niGSCactionParser(self.runcard, self.folder, action)
                parser.build()
                parser.execute(self.format, self.platform)
            # TODO: find a better way to choose between the two parsers
            except (ModuleNotFoundError, KeyError):
                parser = ActionParser(self.runcard, self.folder, action)
                parser.build()
                parser.execute(self.format, self.platform, self.qubits)
                for qubit in self.qubits:
                    if self.platform is not None:
                        self.update_platform_runcard(qubit, action)
            self.dump_report(actions)

        if self.platform is not None:
            self.platform.stop()
            self.platform.disconnect()

    def update_platform_runcard(self, qubit, routine):
        try:
            data_fit = Data.load_data(self.folder, "data", routine, self.format, "fits")
            data_fit.df = data_fit.df[data_fit.df["qubit"] == qubit]
        except FileNotFoundError:
            return None

        params = data_fit.df[data_fit.df["qubit"] == qubit]
        settings = load_yaml(f"{self.folder}/new_platform.yml")
        for param in params:
            if param in list(self.qubits[qubit].__annotations__.keys()):
                setattr(self.qubits[qubit], param, params[param])
                settings["characterization"]["single_qubit"][qubit][param] = int(
                    data_fit.get_values(param)
                )

        with open(f"{self.folder}/new_platform.yml", "w") as file:
            yaml.dump(
                settings, file, sort_keys=False, indent=4, default_flow_style=None
            )

    def dump_report(self, actions=None):
        from qibocal.web.report import create_report

        # update end time
        meta = load_yaml(f"{self.folder}/meta.yml")
        e = datetime.datetime.now(datetime.timezone.utc)
        meta["end-time"] = e.strftime("%H:%M:%S")
        with open(f"{self.folder}/meta.yml", "w") as file:
            yaml.dump(meta, file)

        create_report(self.folder, actions)


class ReportBuilder:
    """Parses routines and plots to report and live plotting page.

    Args:
        path (str): Path to the data folder to generate report for.
        actions (list): List of action to be included in the report. Default is `None`
                        which corresponds to including all the actions in the qq runcard.
    """

    def __init__(self, path, actions=None):
        self.path = path
        self.metadata = load_yaml(os.path.join(path, "meta.yml"))

        # find proper path title
        base, self.title = os.path.join(os.getcwd(), path), ""
        while self.title in ("", "."):
            base, self.title = os.path.split(base)

        self.runcard = load_yaml(os.path.join(path, "runcard.yml"))
        self.format = self.runcard.get("format")
        self.qubits = self.runcard.get("qubits")

        # create calibration routine objects
        # (could be incorporated to :meth:`qibocal.cli.builders.ActionBuilder._build_single_action`)
        self.routines = []
        if actions is None:
            actions = self.runcard.get("actions")

        for action in actions:
            if hasattr(calibrations, action):
                routine = getattr(calibrations, action)
            elif hasattr(calibrations.niGSC, action):
                routine = niGSCactionParser(self.runcard, self.path, action)
                routine.load_plot()
            else:
                raise_error(ValueError, f"Undefined action {action} in report.")

            if not hasattr(routine, "plots"):
                routine.plots = []
            self.routines.append(routine)

    def get_routine_name(self, routine):
        """Prettify routine's name for report headers."""
        return routine.__name__.replace("_", " ").title()

    def get_figure(self, routine, method, qubit):
        """Get html figure for report.

        Args:
            routine (Callable): Calibration method.
            method (Callable): Plot method.
            qubit (int): Qubit id.
        """
        import tempfile

        figures, fitting_report = method(
            self.path, routine.__name__, qubit, self.format
        )
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            html_list = []
            for figure in figures:
                figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
                temp.seek(0)
                fightml = temp.read().decode("utf-8")
                html_list.append(fightml)

            all_html = "".join(html_list)

        return all_html, fitting_report

    def get_live_figure(self, routine, method, qubit):
        """Get url to dash page for live plotting.

        This url is used by :meth:`qibocal.web.app.get_graph`.

        Args:
            routine (Callable): Calibration method.
            method (Callable): Plot method.
            qubit (int): Qubit id.
        """
        return os.path.join(
            method.__name__,
            self.path,
            routine.__name__,
            str(qubit),
            self.format,
        )
