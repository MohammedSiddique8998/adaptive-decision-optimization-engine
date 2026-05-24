from pathlib import Path

import numpy as np


PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]


def _scale(values: np.ndarray, minimum: float, maximum: float, height: int, padding: int) -> np.ndarray:
    span = maximum - minimum
    if span == 0:
        return np.full_like(values, padding + height / 2, dtype=float)
    return padding + height - ((values - minimum) / span) * height


def save_line_chart(series: dict[str, np.ndarray], output_path: Path, title: str, y_label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, padding = 980, 520, 70
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding
    all_values = np.concatenate([values for values in series.values()])
    y_min = float(np.floor(all_values.min() * 10) / 10)
    y_max = float(np.ceil(all_values.max() * 10) / 10)
    if y_min == y_max:
        y_max = y_min + 1.0
    max_len = max(len(values) for values in series.values())
    x = np.linspace(padding, padding + plot_width, max_len)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{padding}" y="34" font-family="Arial" font-size="24" font-weight="700" fill="#111827">{title}</text>',
        f'<line x1="{padding}" y1="{padding + plot_height}" x2="{padding + plot_width}" y2="{padding + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{padding + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<text x="18" y="{padding + plot_height / 2}" font-family="Arial" font-size="14" fill="#374151" transform="rotate(-90 18,{padding + plot_height / 2})">{y_label}</text>',
        f'<text x="{padding + plot_width / 2 - 30}" y="{height - 18}" font-family="Arial" font-size="14" fill="#374151">Step</text>',
    ]

    for tick in range(6):
        y_value = y_min + (y_max - y_min) * tick / 5
        y_pos = padding + plot_height - (plot_height * tick / 5)
        elements.append(f'<line x1="{padding - 5}" y1="{y_pos:.1f}" x2="{padding + plot_width}" y2="{y_pos:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elements.append(f'<text x="8" y="{y_pos + 5:.1f}" font-family="Arial" font-size="12" fill="#4b5563">{y_value:.2f}</text>')

    legend_x, legend_y = padding + plot_width - 180, 54
    for index, (label, values) in enumerate(series.items()):
        color = PALETTE[index % len(PALETTE)]
        y = _scale(values, y_min, y_max, plot_height, padding)
        points = " ".join(f"{x_pos:.1f},{y_pos:.1f}" for x_pos, y_pos in zip(x[: len(values)], y))
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>')
        elements.append(f'<line x1="{legend_x}" y1="{legend_y + index * 22}" x2="{legend_x + 24}" y2="{legend_y + index * 22}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 32}" y="{legend_y + 5 + index * 22}" font-family="Arial" font-size="13" fill="#111827">{label}</text>')

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")
