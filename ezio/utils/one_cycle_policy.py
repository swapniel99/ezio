def ocp(total_it, max_at, min_lr, max_lr):
    lr_lambda = lambda it: iteration(it, max_at, min_lr, max_lr)

    def iteration(it, max_at, min_lr, max_lr):
        if it <= max_at:
            return min_lr + (((max_lr - min_lr) / max_at) * it)

        elif it > max_at:
            return max_lr - (((max_lr - min_lr) / (total_it - max_at)) * (it - max_at))

        else:
            return

    return lr_lambda