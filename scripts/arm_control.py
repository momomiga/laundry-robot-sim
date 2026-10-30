"""Feed-forward on top of the PD position control of the lift and the arm joints (no integral term).

Genesis PD: tau = kp (q_cmd - q) - kv qdot. Two errors of a plain `control_dofs_position(q_plan)` are removed:

  gravity   a static sag of tau_g / kp (lift 55 N / 8000 = 7 mm, joint2 6.6 N m / 300 = 0.022 rad -> about 12 mm at the
            finger bottom). Command q_plan + tau_g(q) / kp. tau_g is the rigid solver's bias force (qf_bias = gravity +
            Coriolis / centrifugal at the current q, qdot; at rest it equals get_dofs_control_force). Contact forces are
            NOT part of it (not an integral term: a blocked arm does not push harder into the cloth or the floor).
            The Coriolis part (a few mN m at our <= 1 rad/s) is left in.
  velocity  the -kv qdot term lags a moving target by kv / kp * qdot (up to 0.145 rad). Command
            q_plan + kv / kp * qdot_plan, i.e. tau = kp (q_plan - q) + kv (qdot_plan - qdot).

Only the lift (index 3) and joint1-6 (4..9) of the 12-vector [base x, y, yaw, lift, joint1-8] are touched (gravity: lift
and arm; velocity: arm only, a lead on the lift gave a 74 mm lag). Base and gripper entries are returned unchanged.
"""
import numpy as np

MODES = ("none", "grav", "grav+vel")
LIFT = 3
ARM = slice(4, 10)


class FeedForward:
    def __init__(self, robot, kp, kv, dt, mode="none", qd_max=3.0, tau_max=None):
        """robot: Genesis entity (joints named as in robot_dof_names()); kp, kv: the PD gains of the 12 dofs;
        qd_max: clip of the plan velocity [rad/s] (a jump of the command must not turn into a spike);
        tau_max: optional clip of tau_g [N m] (default none)."""
        assert mode in MODES, mode
        self.robot, self.mode, self.dt = robot, mode, float(dt)
        self.kp, self.kv = np.asarray(kp, float), np.asarray(kv, float)
        self.qd_max, self.tau_max = qd_max, tau_max
        self.gidx = np.asarray([robot.get_joint(n).dofs_idx[0] for n in robot_dof_names()])
        self._prev = None
        self.last_tau = np.zeros(12)

    def reset(self):
        self._prev = None

    def tau_g(self):
        """Bias force [N, N m] of the 12 dofs at the state of the last step (gravity + Coriolis, no contact)."""
        from genesis.utils.misc import qd_to_torch
        s = self.robot._solver
        b = qd_to_torch(s.dyn_state.dofs.qf_bias, transpose=True).cpu().numpy()
        return np.asarray(b).reshape(-1)[self.gidx].astype(float)

    def command(self, q_plan):
        """PD target for the planned configuration q_plan (12-vector). The plan velocity is the backward difference of
        successive calls (call once per sim step; zero when the plan is held and on the first call)."""
        q_plan = np.asarray(q_plan, float)
        out = q_plan.copy()
        if self.mode == "none":
            return out
        if self.mode == "grav+vel":
            if self._prev is not None:
                qd = np.clip((q_plan - self._prev) / self.dt, -self.qd_max, self.qd_max)
                out[ARM] += self.kv[ARM] / self.kp[ARM] * qd[ARM]
            self._prev = q_plan.copy()
        tau = self.tau_g()
        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)
        self.last_tau = tau
        out[LIFT] += tau[LIFT] / self.kp[LIFT]
        out[ARM] += tau[ARM] / self.kp[ARM]
        return out


def robot_dof_names():
    return ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + [f"joint{i}" for i in range(1, 9)]
