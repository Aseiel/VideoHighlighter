"""
modules/system — this machine — paths, devices, encoders, logging and startup.

Grouping only: these modules were loose at the top of ``modules/``
and import each other by full path as before. Nothing is re-exported
here, so a member is imported as ``modules.system.<name>``.

    app_paths
    compute_backend
    cuda_check
    debug_console
    device_utils
    directml_device
    display_info
    encoder_select
    https_certs
    ort_directml
    repaint_trace
    startup_splash
    ui_scale
"""
