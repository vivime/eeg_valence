
def view(timestamps, streams, inlet, data, window=5, scale=100, refresh=0.2, figure="15x6", version=1, backend='TkAgg'):
    if version == 2:
        import muselsl.viewer_v2
        muselsl.viewer_v2.view()
    else:
        import muselsl.viewer_v1
        muselsl.viewer_v1.view(timestamps, streams, inlet, data, window, scale, refresh, figure, backend)
