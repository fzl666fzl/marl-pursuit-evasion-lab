from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pursuit_lab.constants import PREY_AGENT, PURSUER_AGENTS

PURSUER_COLOR = (220, 60, 60)
PREY_COLOR = (40, 180, 80)
OBSTACLE_COLOR = (120, 120, 120)
BACKGROUND_COLOR = (255, 255, 255)
PANEL_COLOR = (245, 247, 250)


def world_to_pixel(
    position: np.ndarray,
    *,
    size: int,
    top_margin: int,
    padding: int,
    world_extent: float,
) -> tuple[int, int]:
    x = float(np.clip(position[0], -world_extent, world_extent))
    y = float(np.clip(position[1], -world_extent, world_extent))
    plot_left = padding
    plot_right = size - padding
    plot_top = top_margin
    plot_bottom = size - padding
    px = plot_left + int((x + world_extent) / (2 * world_extent) * (plot_right - plot_left))
    py = plot_bottom - int((y + world_extent) / (2 * world_extent) * (plot_bottom - plot_top))
    return px, py


def draw_centered_text(draw: ImageDraw.ImageDraw, center: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((center[0] - width / 2, center[1] - height / 2), text, fill=fill, font=font)


def draw_circle(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color, width=2)
    draw_centered_text(draw, center, label, fill=(255, 255, 255))


def draw_legend(draw: ImageDraw.ImageDraw, size: int) -> None:
    draw.rectangle((0, 0, size, 54), fill=PANEL_COLOR)
    items = [
        (PURSUER_COLOR, "Pursuer / red"),
        (PREY_COLOR, "Prey / green"),
        (OBSTACLE_COLOR, "Obstacle / gray"),
    ]
    x = 18
    for color, label in items:
        draw.ellipse((x, 18, x + 18, 36), fill=color, outline=color)
        draw.text((x + 26, 18), label, fill=(30, 30, 30), font=ImageFont.load_default())
        x += 142


def render_world_frame(env, *, size: int = 720, world_extent: float = 1.25) -> np.ndarray:
    image = Image.new("RGB", (size, size), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    top_margin = 70
    padding = 24
    draw_legend(draw, size)
    draw.rectangle((padding, top_margin, size - padding, size - padding), outline=(210, 210, 210), width=2)

    world = env.unwrapped.world
    for landmark in world.landmarks:
        center = world_to_pixel(
            landmark.state.p_pos,
            size=size,
            top_margin=top_margin,
            padding=padding,
            world_extent=world_extent,
        )
        draw_circle(draw, center, radius=18, color=OBSTACLE_COLOR, label="OBS")

    for agent in world.agents:
        center = world_to_pixel(
            agent.state.p_pos,
            size=size,
            top_margin=top_margin,
            padding=padding,
            world_extent=world_extent,
        )
        if agent.name in PURSUER_AGENTS:
            label = "P" + agent.name.rsplit("_", 1)[-1]
            draw_circle(draw, center, radius=12, color=PURSUER_COLOR, label=label)
        elif agent.name == PREY_AGENT:
            draw_circle(draw, center, radius=12, color=PREY_COLOR, label="E")

    return np.asarray(image, dtype=np.uint8)
