import matplotlib.pyplot as plt
from sim4ad.opendrive import Map, plot_map


class Visualization:

    def __init__(self, statWorld):
        self.statWorld = statWorld
        self.static_world_line_dict = None

    def create_static_world_line_dict(self):
        """
            Creates the multi line data dict for rendering the static world.
            To ensure fast rendering
        """

        self.static_world_line_dict = {
            "xdata": [],
            "ydata": [],
            "color": [],
            "info": []
        }
        if self.statWorld is None:
            return
        static_line_list = self.statWorld.get_lane_marking_dicts(with_centerline=False)
        for lane_markings in static_line_list:
            self.static_world_line_dict["xdata"].append(lane_markings["x_vec"])
            self.static_world_line_dict["ydata"].append(lane_markings["y_vec"])
            self.static_world_line_dict["info"].append("Type: %s" % lane_markings["type"])
            if lane_markings["type"] == "broken":
                self.static_world_line_dict["color"].append("grey")
            else:
                self.static_world_line_dict["color"].append("black")

    def plot_map(self):
        """
        Plot the map using original staticWorld information
        """
        if self.static_world_line_dict is None:
            if self.statWorld is not None:
                self.create_static_world_line_dict()

        line_num = len(self.static_world_line_dict["xdata"])
        for i in range(line_num):
            plt.plot(self.static_world_line_dict["xdata"][i], self.static_world_line_dict["ydata"][i],
                     color=self.static_world_line_dict["color"][i])

        plt.show()

    @staticmethod
    def plot_opendrive_map(opendrive_map):
        """
        Plot the map using opendrive parser
        """
        scenario_map = Map.parse_from_opendrive(opendrive_map)
        plot_map(scenario_map, markings=True, midline=False, drivable=True, plot_background=False)

        plt.show()
