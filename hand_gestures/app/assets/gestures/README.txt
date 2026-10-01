Gesture artwork for the gesture view.

Place one image per model label here, named after the label used by the network
(network/config.py GESTURES), as .svg (preferred) or .png with a transparent or dark background:

    neutral.svg      extension.svg    flexion.svg     ulnar_dev.svg    radial_dev.svg
    grip.svg         abduction.svg    adduction.svg   supination.svg   pronation.svg

Files are picked up automatically (app/assets.py, GESTURE_IMAGES). When a file is missing the app
draws a placeholder hand silhouette with the gesture abbreviation instead.
