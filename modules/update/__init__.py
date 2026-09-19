"""
modules/update — the self-updater.

Grouping only: these modules were loose at the top of ``modules/``
and import each other by full path as before. Nothing is re-exported
here, so a member is imported as ``modules.update.<name>``.

    update_apply
    update_check
    update_download
    update_install
    update_manifest
"""
