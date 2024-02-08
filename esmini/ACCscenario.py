"""
  scenariogeneration
  https://github.com/pyoscx/scenariogeneration

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at https://mozilla.org/MPL/2.0/.

  Copyright (c) 2022 The scenariogeneration Authors.

    An example how to add a sumo controller to an object


    Some features used:

    - Controller

    - Properties

"""
import os.path
import sys

import numpy as np
from tqdm import tqdm

from sim4ad.data import DatasetDataLoader

sys.path.append("./")
print(sys.path)
from scenariogeneration import xosc, ScenarioGenerator
from sim4ad.data.scenario import ScenarioConfig
from sim4ad.opendrive.map import Map


class AccScenario(ScenarioGenerator):
    def __init__(self, episode, map_path):
        super().__init__()
        self.__episode = episode
        self.__map_path = map_path
        self.open_scenario_version = 2

    def scenario(self, **kwargs):
        # create catalogs
        catalog = xosc.Catalog()
        catalog.add_catalog("VehicleCatalog", "../xosc/Catalogs/Vehicles")
        catalog.add_catalog("ControllerCatalog", "../xosc/Catalogs/Controllers")
        # create road
        road = xosc.RoadNetwork(roadfile="../" + self.__map_path)
        scenario_map = Map.parse_from_opendrive(self.__map_path)

        # create parameters
        paramdec = xosc.ParameterDeclarations()
        # create entities
        entities = xosc.Entities()

        # create init
        init = xosc.Init()
        step_time = xosc.TransitionDynamics(xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 1)

        # create the story
        storyparam = xosc.ParameterDeclarations()
        story = xosc.Story("MyStory", storyparam)

        # create controller
        prop = xosc.Properties()

        prop.add_property(name="esminiController", value="ACCController")

        for agent in self.__episode.agents.values():
            if agent.agent_id == 0:
                continue
            # initial RoadId, LaneId and s values
            agent_lane = scenario_map.best_lane_at(agent.state.position, agent.state.heading)
            lane_id = agent_lane.id
            road_id = agent_lane.parent_road.id
            s = agent_lane.distance_at(agent.state.position) + agent_lane.lane_section.start_distance

            ''' ________________________________________ create vehicle ________________________________________ '''
            # make the reference point at the center
            bb = xosc.BoundingBox(width=agent.metadata.width, length=agent.metadata.length,
                                  height=agent.metadata.height, x_center=0.0, y_center=0, z_center=agent.metadata.height/2)
            fa = xosc.Axle(0.523598775598, 0.8, 1.68, 2.98, 0.4)
            ba = xosc.Axle(0.523598775598, 0.8, 1.68, 0, 0.4)
            targetname = "Agent" + str(agent.agent_id)
            ave_speed = np.average(agent.trajectory.velocity)
            '''________________________________________ create ACC entities ________________________________________'''
            # prop.add_property("timeGap", "1.0")
            # prop.add_property("mode", "overwrite")
            # prop.add_property("setSpeed", "%s" % max_speed)
            # prop.add_property("LateralDist", "3")
            # cont = xosc.Controller("ACCController"+targetname, prop)
            cont = xosc.CatalogReference("ControllerCatalog", "ACCController")

            if agent.metadata.agent_type == 'car':
                blue_veh = xosc.Vehicle(
                    "car_blue", xosc.VehicleCategory.car, bb, fa, ba, max_speed=69,
                    max_acceleration=10, max_deceleration=10
                )
                blue_veh.add_property_file("../esmini/resources/models/car_blue.osgb")
                blue_veh.add_property("model_id", "1")
                blue_veh.add_property("scaleMode", "ModelToBB")
                entities.add_scenario_object(targetname, blue_veh, cont)
            elif agent.metadata.agent_type == 'bus' or 'truck':
                # set larger height for truck and bus, dataset has only width and length
                bb = xosc.BoundingBox(width=agent.metadata.width, length=agent.metadata.length,
                                      height=3.5, x_center=0.0, y_center=0,
                                      z_center=3.5 / 2)
                bus_blue = xosc.Vehicle(
                    "bus_blue", xosc.VehicleCategory.bus, bb, fa, ba, max_speed=69,
                    max_acceleration=10, max_deceleration=10
                )
                bus_blue.add_property_file("../esmini/resources/models/bus_blue.osgb")
                bus_blue.add_property("model_id", "6")
                bus_blue.add_property("scaleMode", "ModelToBB")
                entities.add_scenario_object(targetname, bus_blue, cont)
            else:
                raise 'vehicle type is not allowed.'

            ''' ________________________________________ Storyboard Init ________________________________________ '''
            targetspeed = xosc.AbsoluteSpeedAction(ave_speed, step_time)
            targetstart = xosc.TeleportAction(xosc.LanePosition(road_id=road_id, lane_id=lane_id, offset=0, s=s))
            control_action = xosc.ActivateControllerAction(lateral=True, longitudinal=True)
            init.add_init_action(targetname, targetspeed)
            init.add_init_action(targetname, targetstart)
            init.add_init_action(targetname, control_action)
        # create the storyboard
        sb = xosc.StoryBoard(init)

        # create the scenario
        sce = xosc.Scenario(
            name="adapt_speed_control_acc",
            author="C.Wang and F.W. Guo",
            parameters=paramdec,
            entities=entities,
            storyboard=sb,
            roadnetwork=road,
            catalog=catalog,
            osc_minor_version=self.open_scenario_version,
        )
        return sce


def read_data(scenario_name):
    config_path = f"scenarios/configs/{scenario_name}.json"
    data_loader = DatasetDataLoader(config_path)
    data_loader.load()

    episodes = data_loader.scenario.episodes

    episode = episodes[0] # TODO: ARGUMENT TO SELECT EPISODE

    map_path = episode.map_file
    return episode, map_path


def convert_scenario_osc():
    """ read all the extracted scenarios and return their ids"""
    scenario_path = 'scenarios/data/trainingdata/' # TODO: make it a parameter + is it correct it's training data rather than "automatum"?
    scenario_names = []
    for file_name in os.listdir(scenario_path):
        # a scenario is named as 'hw-a9-{scenario_name}-{episode_id}'
        scenario_name = file_name.split('-')[2]
        if scenario_name not in scenario_names:
            scenario_names.append(scenario_name)

    return scenario_names


if __name__ == '__main__':
    scenario_nums = convert_scenario_osc()
    for scenario_num in tqdm(scenario_nums):
        scenario_name = scenario_num + '_scenario'
        episode, map_path = read_data(scenario_name)

        sce = AccScenario(episode, map_path)
        # prettyprint(sce.scenario().get_element())

        sce.basename = scenario_name + '_acc'
        sce.generate('.')
