from diffusers import DiffusionPipeline


def apply_cache_on_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    assert isinstance(pipe, DiffusionPipeline)

    pipe_cls_name = pipe.__class__.__name__
    if pipe_cls_name.startswith("Flux"):
        from .flux import apply_cache_on_pipe as apply_cache_on_pipe_fn
    elif pipe_cls_name.startswith("Sana"):
        from .sana import apply_cache_on_pipe as apply_cache_on_pipe_fn
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")
    return apply_cache_on_pipe_fn(pipe, *args, **kwargs)
