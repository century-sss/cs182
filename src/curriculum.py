import math


class PolyCurriculum:
    def __init__(self, args):
        """
        Unified curriculum controlling:
          - Polynomial degree (task complexity)
          - Number of input points (context length)

        Args:
            args.degree  : has start, end, inc, interval
            args.points  : has start, end, inc, interval
        """
        # 初始化 degree 与 points
        self.max_degree = args.degree.start
        self.n_points = args.points.start

        # 记录 schedule 参数
        self.degree_schedule = args.degree
        self.n_points_schedule = args.points

        # 训练步计数器
        self.step_count = 0

    def update(self):
        """在每个训练 step 调用一次"""
        self.step_count += 1

        self.max_degree = self.update_var(self.max_degree, self.degree_schedule)
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var, schedule):
        """控制单个变量的更新逻辑"""
        if self.step_count % schedule.interval == 0:
            var += schedule.inc
        return min(var, schedule.end)


def get_final_var(init_var, total_steps, inc, n_steps, lim):
    """计算经过若干步 curriculum 后的最终值"""
    final_var = init_var + math.floor(total_steps / n_steps) * inc
    return min(final_var, lim)

class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_points = args.points.start
        self.n_dims_schedule = args.dims
        self.n_points_schedule = args.points
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)
