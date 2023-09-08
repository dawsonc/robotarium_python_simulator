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
    num_agents = 2

    walls = np.array(
        [  # min_x, max_x, min_y, max_y
            [-0.05, 0.05, 1.0, 2.0],
            [-0.05, 0.05, 3.0, 4.0],
            [0.95, 1.05, 2.0, 3.0],
            [0.95, 1.05, 0.0, 1.0],
            [1.95, 2.05, 0.0, 1.0],
            [2.95, 3.05, 0.0, 1.0],
            [2.95, 3.05, 4.0, 5.0],
            [3.95, 4.05, 1.0, 5.0],
            [1.0, 3.0, -0.05, 0.05],
            [0.0, 1.0, 0.95, 1.05],
            [3.0, 4.0, 0.95, 1.05],
            [0.0, 1.0, 1.95, 2.05],
            [0.0, 1.0, 2.95, 3.05],
            [0.0, 3.0, 3.95, 4.05],
            [3.0, 4.0, 4.95, 5.05],
            [2, 3, 2, 3],
        ]
    )
    ingredients = np.array(
        [
            [0.1, 0.9, 1.1, 1.9],
            [0.1, 0.9, 3.1, 3.9],
            [1.1, 1.9, 0.1, 0.9],
            [2.1, 2.9, 0.1, 0.9],
        ]
    )
    kitchen = np.array(
        [
            [3.1, 3.9, 4.1, 4.9],
        ]
    )

    # Adjust for robotarium coordinates
    walls *= 0.4
    walls[:, :2] -= 1.5
    walls[:, 2:] -= 1.0
    ingredients *= 0.4
    ingredients[:, :2] -= 1.5
    ingredients[:, 2:] -= 1.0
    kitchen *= 0.4
    kitchen[:, :2] -= 1.5
    kitchen[:, 2:] -= 1.0

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
        for ingredient in ingredients
    ]
    kitchen_patches = [
        mpatches.Rectangle(
            (kitchen[0], kitchen[2]),
            kitchen[1] - kitchen[0],
            kitchen[3] - kitchen[2],
            facecolor="red",
        )
        for kitchen in kitchen
    ]

    obstacles = wall_patches + ingredient_patches + kitchen_patches

    # Waypoints from Yongchao
    waypoints = np.array(  # (num_agents, num_waypoints, 3), 3 = x, y, time
        [
            [
                [2.500000, 1.500000, 0.000000],
                [2.500000, 0.790000, 0.269206],
                [2.470000, 0.790000, 0.674000],
                [2.530000, 1.420000, 0.812000],
                [1.730000, 1.470000, 0.982000],
                [0.790000, 1.470000, 1.170000],
                [0.790000, 1.530000, 1.182000],
                [2.180000, 1.530000, 1.460000],
                [3.470000, 1.580000, 1.728000],
                [3.470000, 4.210000, 2.254000],
                [3.520000, 4.210000, 2.264000],
            ],
            [
                [1.500000, 2.500000, 0.000000],
                [1.500000, 1.103970, 0.279206],
                [1.500000, 0.790000, 0.342000],
                [1.470000, 0.790000, 0.352000],
                [1.470000, 2.350000, 0.664000],
                [1.470000, 3.470000, 0.888000],
                [0.790000, 3.470000, 1.024000],
                [0.790000, 3.530000, 1.036000],
                [3.470000, 3.530000, 1.572000],
                [3.470000, 4.210000, 1.708000],
                [3.470000, 4.210000 + 0.5, 1.718000],
            ],
        ]
    )
    # Yongchao's initial simulation was 4 m wide and 5 m tall, so adjust to Robotarium
    # dimensions (3 m wide, 2 m tall, so we need to scale down by 2/5)
    waypoints[:, :, :2] *= np.array([0.4, 0.4])
    waypoints[:, :, :2] -= np.array([1.5, 1.0])
    # The initial plan was way too fast, so we need to scale time.
    waypoints[:, :, 2] *= 15.0

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
