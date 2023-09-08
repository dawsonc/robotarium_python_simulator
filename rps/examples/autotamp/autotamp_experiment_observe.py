"""
Define an interface for defining and running AutoTAMP experiments.

This is one really gross big file since we don't get control over the package structure
for the code we upload to robotarium.

Also no type annotations since Robotarium doesn't support them... sadly :(
"""
import matplotlib.patches as mpatches
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.barrier_certificates import (
    create_single_integrator_barrier_certificate_with_boundary,
)
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.transformations import (
    create_si_to_uni_dynamics_with_backwards_motion,
    create_si_to_uni_mapping,
)


class TimedWaypointTrajectory:
    """
    A class to represent a trajectory as a series of time-stamped waypoints.

    Attributes:
        t: An array of timestamps. (N,)
        x: An array of waypoints. (N, dimension)
    """

    def __init__(self, t, x):
        # Sanity check: t and x must have the same length
        if t.shape[0] != x.shape[0]:
            raise ValueError(
                f"t ({t.shape}) and x ({x.shape}) must have the same length."
            )

        # Sanity check: t must be monotonically increasing
        if not np.all(np.diff(t) >= 0):
            raise ValueError(f"t must be monotonically increasing. Got {t}")

        # Sanity check: x must be 2D
        if len(x.shape) != 2:
            raise ValueError(f"x ({x.shape}) must be 2D.")

        self.t = t
        self.x = x

    def at(self, time):
        """
        Return a point along the trajectory, interpolated to the given time.

        Args:
            time: The time to query.

        Returns:
            The point along the trajectory at the given time.
        """
        return np.array(
            [np.interp(time, self.t, self.x[:, i]) for i in range(self.x.shape[-1])]
        )


class AutoTAMPExperiment:
    """A class for running multi-agent planning experiments."""

    def __init__(
        self,
        num_agents,
        initial_agent_poses,
        obstacles,
        agent_plans,
    ):
        """Initialize the expeirment.

        Args:
            num_agents: The number of agents.
            initial_agent_poses: The initial poses of the agents (x, y, theta).
                (num_agents, 3)
            obstacles: A list of matplotlib patches representing obstacles.
            agent_plans: A list of TimedWaypointTrajectory objects, one for each agent.
        """
        # Sanity check: num_agents must be positive, match the number of agent plans,
        # and match the number of initial agent positions.
        if num_agents <= 0:
            raise ValueError(f"num_agents ({num_agents}) must be positive.")
        if num_agents != len(agent_plans):
            raise ValueError(
                (
                    f"num_agents ({num_agents}) must match the "
                    f"number of agent plans ({len(agent_plans)})."
                )
            )
        if num_agents != initial_agent_poses.shape[0]:
            raise ValueError(
                (
                    f"num_agents ({num_agents}) must match the number of "
                    f"initial agent positions ({initial_agent_poses.shape[0]})."
                )
            )

        self.num_agents = num_agents
        self.initial_agent_poses = initial_agent_poses
        self.obstacles = obstacles
        self.agent_plans = agent_plans

    def run(
        self,
        experiment_time,
        display_plot=False,
        show_trajectories=False,
        starting_position_tolerance=0.1,
    ):
        """Run the experiment.

        Args:
            experiment_time: The time to run the experiment for.
            display_plot: Whether to plot the experiment.
            show_trajectories: Whether to show the trajectories of the agents.
            starting_position_tolerance: The tolerance for starting positions.
        """

        #####################
        # Experiment Set Up #
        #####################

        # Create the robotarium object
        r = robotarium.Robotarium(
            number_of_robots=self.num_agents,
            show_figure=display_plot,
            initial_conditions=self.initial_agent_poses.T,
            sim_in_real_time=False,
        )

        # Create single integrator position controller
        single_integrator_position_controller = create_si_position_controller(
            x_velocity_gain=2.0, y_velocity_gain=2.0
        )

        # Create barrier certificates to avoid collision
        si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

        _, uni_to_si_states = create_si_to_uni_mapping()

        # Create mapping from single integrator velocity commands to unicycle commands
        si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

        # Set up plotting
        for obstacle in self.obstacles:
            r.axes.add_patch(obstacle)

        if show_trajectories:
            for plan in self.agent_plans:
                r.axes.plot(plan.x[:, 0], plan.x[:, 1], "k--o")

        goal_markers = [
            r.axes.scatter(plan.at(0.0)[0], plan.at(0.0)[1], c="r")
            for plan in self.agent_plans
        ]

        # Get the initial robot states
        x = r.get_poses()
        x_si = uni_to_si_states(x)

        ########################
        # Move to start points #
        ########################
        r.step()
        distance_to_start = np.linalg.norm(
            x_si[:2, :].T - self.initial_agent_poses[:, :2], axis=1
        )
        print("Moving to start points...")
        while np.any(distance_to_start > starting_position_tolerance):
            print(distance_to_start)
            # Get the current poses of the robots
            x = r.get_poses()
            x_si = uni_to_si_states(x)

            # Get the desired poses of the robots
            x_si_desired = self.initial_agent_poses[:, :2].T

            # Create single-integrator control inputs to track the desired poses
            dxi = single_integrator_position_controller(x_si, x_si_desired)

            # Create safe control inputs (i.e., no collisions)
            dxi = si_barrier_cert(dxi, x_si)

            # Transform the single-integrator inputs to unicycle inputs
            dxu = si_to_uni_dyn(dxi, x)

            # Set the velocities of the robots
            r.set_velocities(np.arange(self.num_agents), dxu)

            r.step()
            distance_to_start = np.linalg.norm(
                x_si[:2, :].T - self.initial_agent_poses[:, :2], axis=1
            )

        ######################
        # Run the experiment #
        ######################
        current_time = 0.0
        print("Reached start points. Running experiment...")

        # Simulate for the desired amount of time
        while current_time < experiment_time:
            # Get the current poses of the robots
            x = r.get_poses()
            x_si = uni_to_si_states(x)

            # Get the desired poses of the robots
            x_si_desired = np.array(
                [plan.at(current_time) for plan in self.agent_plans]
            ).T

            if show_trajectories:
                for plan, goal_marker in zip(self.agent_plans, goal_markers):
                    goal_marker.set_offsets(plan.at(current_time)[:2])

            # Create single-integrator control inputs to track the desired poses
            dxi = single_integrator_position_controller(x_si, x_si_desired)

            # Create safe control inputs (i.e., no collisions)
            dxi = si_barrier_cert(dxi, x_si)

            # Transform the single-integrator inputs to unicycle inputs
            dxu = si_to_uni_dyn(dxi, x)

            # Set the velocities of the robots
            r.set_velocities(np.arange(self.num_agents), dxu)

            current_time += r.time_step
            r.step()

        # Call at end of script to print debug information to run on the Robotarium
        # server properly
        r.call_at_scripts_end()


if __name__ == "__main__":
    # Define an example experiment.
    num_agents = 3

    walls = np.array(
        [  # min_x, max_x, min_y, max_y
            [-0.1, 0.1, 0.0, 11.0],
            [11.9, 12.1, 0.0, 11.0],
            [0.0, 12.0, -0.1, 0.1],
            [0.0, 12.0, 10.9, 11.1],
            [2.9, 3.1, 3.0, 4.0],
            [2.9, 3.1, 7.0, 8.0],
            [8.9, 9.1, 3.0, 4.0],
            [8.9, 9.1, 7.0, 8.0],
            [5.9, 6.1, 2.5, 8.5],
            [5.0, 7.0, 2.4, 2.6],
            [5.0, 7.0, 8.4, 8.6],
        ]
    )
    observation_regions = np.array(
        [
            [0.8, 2.2, 0.8, 2.2],
            [0.8, 2.2, 8.8, 10.2],
            [9.8, 11.2, 0.8, 2.2],
            [9.8, 11.2, 8.8, 10.2],
            [5.3, 6.7, 9.3, 10.7],
            [5.3, 6.7, 0.3, 1.7],
        ]
    )
    transmitters = np.array(
        [
            [0.8, 2.2, 4.8, 6.2],
            [9.8, 11.2, 4.8, 6.2],
        ]
    )
    start_areas = np.array(
        [
            [4.0, 8.0, 4.0, 7.0],
        ]
    )

    # Adjust for robotarium coordinates. Sim coordinates are 11 m square, so we need to
    # scale down by 2 / 11
    walls *= 2 / 11.0
    walls[:, :2] -= 1.5
    walls[:, 2:] -= 1.0
    observation_regions *= 2 / 11.0
    observation_regions[:, :2] -= 1.5
    observation_regions[:, 2:] -= 1.0
    transmitters *= 2 / 11.0
    transmitters[:, :2] -= 1.5
    transmitters[:, 2:] -= 1.0
    start_areas *= 2 / 11.0
    start_areas[:, :2] -= 1.5
    start_areas[:, 2:] -= 1.0

    wall_patches = [
        mpatches.Rectangle(
            (wall[0], wall[2]), wall[1] - wall[0], wall[3] - wall[2], facecolor="black"
        )
        for wall in walls
    ]
    ingredient_patches = [
        mpatches.Rectangle(
            (ingredient[0], ingredient[2]),
            ingredient[1] - ingredient[0],
            ingredient[3] - ingredient[2],
            facecolor="blue",
        )
        for ingredient in observation_regions
    ]
    transmitters_patches = [
        mpatches.Rectangle(
            (transmitters[0], transmitters[2]),
            transmitters[1] - transmitters[0],
            transmitters[3] - transmitters[2],
            facecolor="red",
        )
        for transmitters in transmitters
    ]
    start_area_patches = [
        mpatches.Rectangle(
            (start_area[0], start_area[2]),
            start_area[1] - start_area[0],
            start_area[3] - start_area[2],
            facecolor="pink",
        )
        for start_area in start_areas
    ]

    obstacles = (
        ingredient_patches + transmitters_patches + start_area_patches + wall_patches
    )

    # Waypoints from Yongchao
    waypoints = np.array(  # (num_agents, num_waypoints, 3), 3 = x, y, time
        [
            [
                [7.500000, 6.500000, 0.000000],
                [7.500000, 8.910000, 0.426000],
                [9.910000, 8.910000, 0.90000],
                [9.910000, 6.090000, 1.600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],
                [7.500000, 6.090000, 2.2600000],  # pad to same length
            ],
            [
                [4.500000, 4.500000, 0.000000],
                [4.500000, 2.140000, 0.472000],
                [2.090000, 2.090000, 0.964000],
                [2.090000, 1.530000, 1.076000],
                [5.410000, 1.530000, 1.740000],
                [5.460000, 1.530000, 1.750000],
                [9.910000, 1.530000, 2.640000],
                [9.910000, 2.090000, 2.752000],
                [9.910000, 4.89000, 3.316000],
                [7.50000, 4.89000, 3.816000],
                [7.50000, 4.89000, 3.816000],
                [7.50000, 4.89000, 3.816000],
                [7.50000, 4.89000, 3.816000],
                [7.50000, 4.89000, 3.816000],
                [7.50000, 4.89000, 3.816000],
                [7.50000, 4.89000, 3.816000],  # pad to same length
            ],
            [
                [4.500000, 6.500000, 0.000000],
                [4.500000, 6.555000, 0.011000],
                [4.500000, 6.605000, 0.021000],
                [4.500000, 6.820000, 0.064000],
                [4.500000, 6.870000, 0.074000],
                [4.530000, 6.890000, 0.084000],
                [4.580000, 6.890000, 0.094000],
                [4.580000, 9.470000, 0.610000],
                [5.410000, 9.470000, 0.776000],
                [5.410000, 9.470000, 0.786000],
                [2.090000, 9.470000, 1.450000],
                [2.090000, 9.420000, 1.460000],
                [2.090000, 6.090000, 2.126000],
                [2.090000, 6.090000, 2.136000],
                [4.110000, 6.090000, 2.540000],
                [4.110000, 6.058916, 2.550000],
            ],
        ]
    )
    # Yongchao's initial simulation was 4 m wide and 5 m tall, so adjust to Robotarium
    # dimensions (3 m wide, 2 m tall, so we need to scale down by 2/5)
    waypoints[:, :, :2] *= np.array([2 / 11.0, 2 / 11.0])
    waypoints[:, :, :2] -= np.array([1.5, 1.0])
    # The initial plan was way too fast, so we need to scale time.
    waypoints[:, :, 2] *= 10.0

    agent_plans = [
        TimedWaypointTrajectory(waypoints[i, :, 2], waypoints[i, :, :2])
        for i in range(num_agents)
    ]
    initial_agent_poses = np.hstack(  # start facing upwards (pi/2)
        (waypoints[:num_agents, 0, :2], np.zeros((num_agents, 1)) + np.pi / 2)
    )

    # Create the experiment
    experiment = AutoTAMPExperiment(
        num_agents, initial_agent_poses, obstacles, agent_plans
    )

    # Run the experiment
    experiment.run(
        waypoints[:, :, 2].max() + 1.0, display_plot=True, show_trajectories=False
    )
