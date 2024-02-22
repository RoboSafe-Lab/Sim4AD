"""
  scenariogeneration
  https://github.com/pyoscx/scenariogeneration

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at https://mozilla.org/MPL/2.0/.

  Copyright (c) 2022 The scenariogeneration Authors.

    An example showing how to create a trajectory based on polyline
    Also shows how to create a vehicle from start

    Some features used:

    - RelativeLanePosition

    - Polyline

    - Trajectory

    - TimeHeadwayCondition

    - FollowTrajectoryAction

"""
import os

import numpy as np
from tqdm import tqdm
from scenariogeneration import xosc, prettyprint, ScenarioGenerator
from ACCscenario import read_data, convert_scenario_osc


class TrjScenario(ScenarioGenerator):
    def __init__(self, episode, map_path):
        super().__init__()
        self.__episode = episode
        self.__map_path = map_path
        self.open_scenario_version = 2

    def scenario(self, **kwargs):
        ### create catalogs
        catalog = xosc.Catalog()
        catalog.add_catalog("VehicleCatalog", "../xosc/Catalogs/Vehicles")

        ### create road
        road = xosc.RoadNetwork(roadfile="" + self.__map_path)

        ### create parameters
        paramdec = xosc.ParameterDeclarations()
        ## create entities
        entities = xosc.Entities()
        ### create init
        init = xosc.Init()
        step_time = xosc.TransitionDynamics(
            xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 0
        )

        ## create the story
        storyparam = xosc.ParameterDeclarations()
        story = xosc.Story("mystory", storyparam)

        for inx, agent in enumerate(self.__episode.agents.values()):

            positionlist = []
            for i in range(len(agent.x_vec)):
                pos = (agent.x_vec[i], agent.y_vec[i])
                heading = agent.psi_vec[i]
                positionlist.append(xosc.WorldPosition(x=pos[0], y=pos[1], z=None, h=heading, p=None, r=None))
            time = agent.time
            polyline = xosc.Polyline(time, positionlist)

            bb = xosc.BoundingBox(width=agent.width, length=agent.length,
                                  height=2, x_center=2.0, y_center=0, z_center=0.9) # TODO: height is hardcoded
            fa = xosc.Axle(0.523598775598, 0.8, 1.68, 2.98, 0.4)
            ba = xosc.Axle(0.523598775598, 0.8, 1.68, 0, 0.4)
            targetname = "Agent " + str(agent.UUID)

            if agent.type == 'car':
                blue_veh = xosc.Vehicle(
                    "car_blue", xosc.VehicleCategory.car, bb, fa, ba, max_speed=69,
                    max_acceleration=10, max_deceleration=10
                )
                blue_veh.add_property_file("../esmini/resources/models/car_blue.osgb")
                blue_veh.add_property("model_id", "1")
                entities.add_scenario_object(targetname, blue_veh)
            elif agent.type  == 'truck':
                truck_yellow = xosc.Vehicle(
                    "truck_yellow", xosc.VehicleCategory.truck, bb, fa, ba, max_speed=69,
                    max_acceleration=10, max_deceleration=10
                )
                truck_yellow.add_property_file("../esmini/resources/models/truck_yellow.osgb")
                truck_yellow.add_property("model_id", "4")
                entities.add_scenario_object(targetname, truck_yellow)
            elif agent.type  == 'bus':
                bus_blue = xosc.Vehicle(
                    "bus_blue", xosc.VehicleCategory.bus, bb, fa, ba, max_speed=69,
                    max_acceleration=10, max_deceleration=10
                )
                bus_blue.add_property_file("../esmini/resources/models/bus_blue.osgb")
                bus_blue.add_property("model_id", "6")
                entities.add_scenario_object(targetname, bus_blue)
            else:
                raise f'vehicle type {agent.type } is not allowed.'

            # Since we only have the x and y velocity, we want to compute the total velocity using the Pythagorean theorem
            velocities = np.sqrt(np.array(agent.vx_vec)**2 + np.array(agent.vy_vec)**2)

            #max_speed = max(velocities)
            #targetspeed = xosc.AbsoluteSpeedAction(max_speed, step_time)
            initial_pos = (agent.x_vec[0], agent.y_vec[0])
            initial_heading = agent.psi_vec[0]
            targetstart = xosc.TeleportAction(xosc.WorldPosition(x=initial_pos[0], y=initial_pos[1], z=None,
                                                                  h=initial_heading, p=None, r=None))

            # init.add_init_action(targetname, targetspeed)
            init.add_init_action(targetname, targetstart)
            init.add_init_action(targetname, xosc.VisibilityAction(False, False, False))

            traj = xosc.Trajectory("trajectory" + targetname, False)
            traj.add_shape(polyline)

            trajact = xosc.FollowTrajectoryAction(
                traj, xosc.FollowingMode.follow, xosc.ReferenceContext.absolute, 1, 0
            )

            # init.add_init_action(targetname, trajact)

            # create the start trigger which will spawn the vehicle when it first appears in the scene
            starttrigger = xosc.ValueTrigger(name="spawn_trigger", delay=0, conditionedge=xosc.ConditionEdge.none,
                                             valuecondition=xosc.SimulationTimeCondition(time[0],
                                                                                         xosc.Rule.greaterThan))

            ### create an event
            event = xosc.Event("Start" + targetname, xosc.Priority.override)
            event.add_trigger(starttrigger)

            ## create the act
            man = xosc.Maneuver("man" + targetname)
            mangr = xosc.ManeuverGroup("mangroup" + targetname)

            event.add_action("Action" + targetname, trajact)
            event.add_action("Become Visible", xosc.VisibilityAction(True, True, True))
            man.add_event(event)
            mangr.add_actor(targetname)
            mangr.add_maneuver(man)

            # create the end trigger which will destroy the vehicle when its time is up
            endtrigger = xosc.ValueTrigger(name="destroy_trigger", delay=0, conditionedge=xosc.ConditionEdge.none,
                                           valuecondition=xosc.SimulationTimeCondition(agent.time[-1],
                                                                                       xosc.Rule.greaterThan),
                                           triggeringpoint="stop")
            
            act = xosc.Act("Behavior" + targetname, starttrigger=starttrigger)
            act.add_maneuver_group(mangr)

            story.add_act(act)

        ## create the storyboard
        sb = xosc.StoryBoard(init)
        sb.add_story(story)

        ## create the scenario
        sce = xosc.Scenario(
            name="trajectory_reproduction",
            author="C.Wang and F.W. Guo",
            parameters=paramdec,
            entities=entities,
            storyboard=sb,
            roadnetwork=road,
            catalog=catalog,
            osc_minor_version=self.open_scenario_version,
        )
        return sce


"""
The code will convert a specific episode to the OpenSCenario (OSC) format and display the trajectories using esmini.
"""
if __name__ == "__main__":
    scenario_names = convert_scenario_osc()

    for scenario_name in tqdm(scenario_names):
        episode, map_path = read_data(scenario_name)

        sce = TrjScenario(episode, map_path)
        # prettyprint(sce.scenario().get_element())

        sce.basename = scenario_name
        sce.generate('scenarios')

        # uncomment the following lines to display the scenario using esmini
        additional_args = ""
        window_size = "60 60 800 400"
        additional_args += " --window " + window_size
        xosc_name = scenario_name + "0" + ".xosc"
        additional_args += " --osc " + "scenarios/xosc/" + xosc_name
        directory = os.getcwd()
        executable_path = "/Users/massimiliano/Library/CloudStorage/OneDrive-UniversityofEdinburgh/A-Uni2324/MInf/Sim4AD/esmini/esmini/bin/esmini" # TODO
        os.system(executable_path+additional_args)
