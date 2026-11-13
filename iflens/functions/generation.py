# Prompt that asks for atmosphere/lighting changes only and protects the objects.
# The old prompt ("Without making any other significant composition changes...") did not
# forbid moving, resizing or redrawing objects, which breaks bounding-box annotations.
BASE_GENERATION_PROMPT = (
    "Edit the attached photo by changing ONLY the atmospheric and lighting conditions "
    "described below. Keep every vessel, structure, buoy and other object exactly where it is: "
    "same position, size, shape, orientation, colour and count. Do not add, remove, move, "
    "resize or redraw any object. Do not crop, rotate or change the framing, aspect ratio, "
    "camera viewpoint or horizon line. Conditions to apply:\n"
)
