"""Generate the frozen 16-item G-PC0 procedural mechanism-control bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "evaluation"))

from g_pc0 import GENERATOR_ID, PROFILE, SOURCE_DATASET, validate_g_pc0_assets, validate_g_pc0_manifest


SEED = 20260914
CANVAS = (720, 520)
FONT_PATH = Path("/usr/share/fonts/dejavu/DejaVuSans.ttf")
BOLD_FONT_PATH = Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")
GENERIC_CONTEXT = (
    "Inspect the visible rules in order. Track each intermediate state carefully, compare it with the "
    "next displayed relation, and verify the final state against all four choices using only evidence "
    "shown in the image. Check the ordered transitions twice before selecting the matching choice."
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = BOLD_FONT_PATH if bold else FONT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"frozen font is missing: {path}")
    return ImageFont.truetype(str(path), size=size)


def base_image(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", CANVAS, "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((12, 12, 708, 508), radius=18, fill="#ffffff", outline="#334155", width=3)
    draw.text((36, 28), title, fill="#0f172a", font=font(28, True))
    draw.text((36, 66), subtitle, fill="#475569", font=font(16))
    return image, draw


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
    draw.line((start, end), fill=color, width=6)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 14
    left = (
        end[0] - size * math.cos(angle - math.pi / 6),
        end[1] - size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - size * math.cos(angle + math.pi / 6),
        end[1] - size * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left, right), fill=color)


def draw_node_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    node_radius: int = 32,
) -> None:
    """Stop an arrow outside node circles so its direction remains visible."""

    dx, dy = end[0] - start[0], end[1] - start[1]
    distance = math.hypot(dx, dy)
    unit = (dx / distance, dy / distance)
    clipped_start = (int(start[0] + unit[0] * node_radius), int(start[1] + unit[1] * node_radius))
    clipped_end = (int(end[0] - unit[0] * node_radius), int(end[1] - unit[1] * node_radius))
    draw_arrow(draw, clipped_start, clipped_end, color)


def place_options(correct: str, distractors: list[str], correct_label: str) -> dict[str, str]:
    if len(distractors) != 3 or len({correct.casefold(), *(value.casefold() for value in distractors)}) != 4:
        raise ValueError("options must be four distinct values")
    choices = list("ABCD")
    output: dict[str, str] = {correct_label: correct}
    for label, value in zip((label for label in choices if label != correct_label), distractors):
        output[label] = value
    return {label: output[label] for label in choices}


def chain_graph(evidence: list[str], states: list[str], answer_claim: str) -> tuple[dict, int]:
    """Build a chain in which each new state also depends on the next visible rule."""

    nodes = [
        {"id": f"e{index}", "type": "evidence", "claim": claim}
        for index, claim in enumerate(evidence, start=1)
    ]
    nodes.extend(
        {"id": f"i{index}", "type": "intermediate", "claim": claim}
        for index, claim in enumerate(states, start=1)
    )
    nodes.append({"id": "a", "type": "answer", "claim": answer_claim})
    edges: list[list[str]] = []
    for index in range(1, len(states) + 1):
        target = f"i{index}"
        edges.append([f"e{index}", target])
        if index > 1:
            edges.append([f"i{index - 1}", target])
    final_source = f"i{len(states)}"
    final_evidence = len(states) + 1
    if final_evidence <= len(evidence):
        edges.append([f"e{final_evidence}", "a"])
    edges.append([final_source, "a"])
    for index in range(final_evidence + 1, len(evidence) + 1):
        edges.append([f"e{index}", "a"])
    return {"nodes": nodes, "edges": edges}, len(states) + 1


def common_record(
    *,
    global_index: int,
    family: str,
    image_path: str,
    image_checksum: str,
    question: str,
    options: dict[str, str],
    correct_label: str,
    evidence: list[str],
    graph: dict,
    depth: int,
    guides: list[dict],
    solution_trace: list[str],
    spec_hash: str,
) -> dict:
    record_id = f"gpc0-{global_index + 1:02d}"
    return {
        "sample_id": record_id,
        "source_dataset": SOURCE_DATASET,
        "source_record_id": record_id,
        "source_image_id": f"gpc0-image-{global_index + 1:02d}",
        "source_cluster_id": f"gpc0-cluster-{global_index + 1:02d}",
        "image_original": image_path,
        "image_checksum": image_checksum,
        "final_question": question,
        **{f"option_{label}": options[label] for label in "ABCD"},
        "final_answer": correct_label,
        "canonical_answer_text": options[correct_label],
        "generic_context": GENERIC_CONTEXT,
        "knowledge_dependency_label": "image_only",
        "task_family": "R",
        "full_image_reference_valid": True,
        "unique_answer": True,
        "human_verified": False,
        "human_reviewer": "",
        "human_reviewed_at": "",
        "procedural_gold_verified": True,
        "answer_leakage": 0,
        "external_fact_injected": 0,
        "required_evidence": evidence,
        "difficult_visual_evidence": [],
        "reasoning_depth": depth,
        "minimal_reasoning_graph": graph,
        "guides": guides,
        "solution_trace": solution_trace,
        "manifest_profile": PROFILE,
        "experiment_role": "mechanism_positive_control",
        "positive_control_family": family,
        "generator_id": GENERATOR_ID,
        "generator_seed": SEED,
        "generation_spec_hash": spec_hash,
        "eligible_for_natural_g0": False,
        "eligible_for_training": False,
    }


PATH_SPECS = (
    ("J", "K", "M", "R", "V", ("red", "blue", "green", "orange")),
    ("B", "F", "H", "N", "T", ("blue", "orange", "red", "green")),
    ("C", "G", "L", "Q", "W", ("green", "red", "orange", "blue")),
    ("D", "I", "P", "S", "X", ("orange", "green", "blue", "red")),
)
COLOR_MAP = {"red": "#dc2626", "blue": "#2563eb", "green": "#16a34a", "orange": "#ea580c"}


def render_directed_path(index: int, path: Path, rng: random.Random) -> dict:
    start, first, second, third, final, colors = PATH_SPECS[index]
    labels = [start, first, second, third, final] + [value for value in "ACEGILNQZ" if value not in {start, first, second, third, final}][:4]
    positions = [(120, 160), (330, 140), (560, 175), (550, 350), (320, 390), (110, 350), (220, 260), (430, 255), (640, 290)]
    image, draw = base_image("Directed path", "Follow the requested edge colors from START.")
    for step in range(4):
        draw_node_arrow(draw, positions[step], positions[step + 1], COLOR_MAP[colors[step]])
    decoy_pairs = [(5, 6, "blue"), (6, 7, "orange"), (7, 8, "green"), (8, 5, "red")]
    for left, right, color in decoy_pairs:
        draw_node_arrow(draw, positions[left], positions[right], COLOR_MAP[color])
    for node_index, (x, y) in enumerate(positions):
        radius = 28
        fill = "#fef3c7" if node_index == 0 else "#e2e8f0"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline="#0f172a", width=3)
        draw.text((x, y), labels[node_index], anchor="mm", fill="#0f172a", font=font(22, True))
    draw.text((76, 105), "START", fill="#92400e", font=font(16, True))
    draw.text((36, 474), "Edge order is given in the question; arrow direction matters.", fill="#475569", font=font(15))
    image.save(path, format="PNG", optimize=False)

    sequence = ", ".join(colors)
    answer = final
    distractors = [value for value in labels[5:] if value not in {first, second, third}][:3]
    evidence = [
        f"The {colors[0]} arrow from {start} ends at {first}",
        f"The {colors[1]} arrow from {first} ends at {second}",
        f"The {colors[2]} arrow from {second} ends at {third}",
        f"The {colors[3]} arrow from {third} ends at {final}",
    ]
    states = [f"After step 1 the current node is {first}", f"After step 2 the current node is {second}", f"After step 3 the current node is {third}"]
    graph, depth = chain_graph(evidence, states, f"The final node is {final}")
    guides = [
        {"question": f"Starting at {start}, where does the {colors[0]} arrow lead?", "answer": first, "supports_next": "i1"},
        {"question": f"From that node, where does the {colors[1]} arrow lead?", "answer": second, "supports_next": "i2"},
        {"question": f"From that node, where does the {colors[2]} arrow lead?", "answer": third, "supports_next": "i3"},
    ]
    return {
        "question": f"Begin at node {start}. Follow arrow colors in this order: {sequence}. Which node do you reach?",
        "answer": answer,
        "distractors": distractors,
        "evidence": evidence,
        "graph": graph,
        "depth": depth,
        "guides": guides,
        "solution_trace": [start, first, second, third, final],
    }


TRANSFORM_SPECS = (
    (3, (("+", 4), ("x", 2), ("-", 5), ("+", 6)), (11, 18, 21)),
    (5, (("x", 3), ("-", 4), ("+", 7), ("/", 2)), (8, 12, 19)),
    (8, (("-", 3), ("x", 4), ("+", 2), ("/", 2)), (9, 13, 24)),
    (6, (("+", 5), ("x", 2), ("-", 8), ("+", 3)), (12, 16, 20)),
)


def apply_op(value: int, operation: tuple[str, int]) -> int:
    symbol, amount = operation
    if symbol == "+":
        return value + amount
    if symbol == "-":
        return value - amount
    if symbol == "x":
        return value * amount
    if symbol == "/" and value % amount == 0:
        return value // amount
    raise ValueError("operation does not preserve integers")


def render_sequential_transform(index: int, path: Path, rng: random.Random) -> dict:
    start, operations, distractors = TRANSFORM_SPECS[index]
    values = [start]
    for operation in operations:
        values.append(apply_op(values[-1], operation))
    image, draw = base_image("Sequential transform", "Apply each displayed operation from left to right.")
    xs = [72, 215, 358, 501, 644]
    for position, value in enumerate([str(start), *(f"{symbol}{amount}" for symbol, amount in operations)]):
        x = xs[position]
        fill = "#fef3c7" if position == 0 else "#dbeafe"
        draw.rounded_rectangle((x - 50, 205, x + 50, 285), radius=12, fill=fill, outline="#1e3a8a", width=3)
        draw.text((x, 245), value, anchor="mm", fill="#0f172a", font=font(27, True))
        if position < 4:
            draw_arrow(draw, (x + 54, 245), (xs[position + 1] - 56, 245), "#64748b")
    draw.text((34, 155), "START", fill="#92400e", font=font(16, True))
    draw.text((36, 340), "All values stay whole numbers. The final box is an operation, not a displayed result.", fill="#475569", font=font(14))
    image.save(path, format="PNG", optimize=False)

    evidence = [f"The start value is {start}"] + [
        f"Operation {step} is {symbol}{amount}" for step, (symbol, amount) in enumerate(operations, start=1)
    ]
    states = [f"After operation {step} the value is {values[step]}" for step in range(1, 4)]
    # The first state depends on both the start and operation 1; encode those as roots.
    graph, depth = chain_graph(evidence[1:], states, f"After operation 4 the value is {values[4]}")
    graph["nodes"].insert(0, {"id": "e0", "type": "evidence", "claim": evidence[0]})
    graph["edges"].append(["e0", "i1"])
    evidence_for_graph = evidence
    guides = [
        {"question": f"What value results after applying {operations[0][0]}{operations[0][1]} to the start value?", "answer": str(values[1]), "supports_next": "i1"},
        {"question": "What is the value after the second displayed operation?", "answer": str(values[2]), "supports_next": "i2"},
        {"question": "What is the value after the third displayed operation?", "answer": str(values[3]), "supports_next": "i3"},
    ]
    return {
        "question": "Starting with the number in the START box, apply all four displayed operations from left to right. What is the final value?",
        "answer": str(values[4]),
        "distractors": [str(value) for value in distractors],
        "evidence": evidence_for_graph,
        "graph": graph,
        "depth": depth,
        "guides": guides,
        "solution_trace": [str(value) for value in values],
    }


GRID_SPECS = (
    (((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)), "AMBER", ("COBALT", "IVORY", "JADE")),
    (((4, 0), (3, 0), (3, 1), (2, 1), (2, 2)), "COBALT", ("AMBER", "JADE", "IVORY")),
    (((0, 4), (1, 4), (1, 3), (2, 3), (3, 3)), "IVORY", ("JADE", "AMBER", "COBALT")),
    (((4, 4), (4, 3), (3, 3), (3, 2), (2, 2)), "JADE", ("IVORY", "COBALT", "AMBER")),
)
DIR_SYMBOL = {(0, 1): ">", (0, -1): "<", (1, 0): "v", (-1, 0): "^"}


def grid_position(value: tuple[int, int]) -> str:
    return f"row {value[0] + 1}, column {value[1] + 1}"


def render_grid_trace(index: int, path: Path, rng: random.Random) -> dict:
    route, answer, distractors = GRID_SPECS[index]
    image, draw = base_image("Grid trace", "Move one cell in the direction shown at each visited cell.")
    left, top, cell = 135, 120, 66
    for row in range(5):
        for column in range(5):
            x0, y0 = left + column * cell, top + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill="#f8fafc", outline="#94a3b8", width=2)
    for index_step in range(4):
        current, following = route[index_step], route[index_step + 1]
        delta = (following[0] - current[0], following[1] - current[1])
        x = left + current[1] * cell + cell // 2
        y = top + current[0] * cell + cell // 2
        draw.text((x, y), DIR_SYMBOL[delta], anchor="mm", fill="#1d4ed8", font=font(31, True))
    start = route[0]
    sx, sy = left + start[1] * cell + 8, top + start[0] * cell + 7
    draw.ellipse((sx, sy, sx + 13, sy + 13), fill="#f59e0b", outline="#92400e")
    tokens = [(route[-1], answer), ((0, 2), distractors[0]), ((4, 2), distractors[1]), ((2, 4), distractors[2])]
    for (row, column), token in tokens:
        x = left + column * cell + cell // 2
        y = top + row * cell + cell // 2
        draw.text((x, y + 18), token[:3], anchor="mm", fill="#7c3aed", font=font(13, True))
    for number in range(5):
        draw.text((left - 22, top + number * cell + cell // 2), str(number + 1), anchor="mm", fill="#475569", font=font(14))
        draw.text((left + number * cell + cell // 2, top - 18), str(number + 1), anchor="mm", fill="#475569", font=font(14))
    draw.text((500, 145), "TOKEN KEY", fill="#0f172a", font=font(16, True))
    for token_index, token in enumerate(("AMBER", "COBALT", "IVORY", "JADE")):
        draw.text((500, 182 + token_index * 34), f"{token[:3]} = {token}", fill="#475569", font=font(15))
    draw.text((36, 474), "The gold dot marks START. Read the token only after four moves.", fill="#475569", font=font(15))
    image.save(path, format="PNG", optimize=False)

    evidence = []
    for step in range(4):
        current, following = route[step], route[step + 1]
        evidence.append(f"The arrow at {grid_position(current)} leads to {grid_position(following)}")
    evidence.append(f"The final cell at {grid_position(route[-1])} contains token {answer}")
    states = [f"After move {step} the position is {grid_position(route[step])}" for step in range(1, 5)]
    graph, depth = chain_graph(evidence, states, f"The final token is {answer}")
    # chain_graph attached e5 directly to answer; i4 records the final position.
    guides = [
        {"question": f"From START, where does the first arrow move?", "answer": grid_position(route[1]), "supports_next": "i1"},
        {"question": "Where are you after the second move?", "answer": grid_position(route[2]), "supports_next": "i2"},
        {"question": "Where are you after the third move?", "answer": grid_position(route[3]), "supports_next": "i3"},
        {"question": "Where are you after the fourth move, before reading its token?", "answer": grid_position(route[4]), "supports_next": "i4"},
    ]
    return {
        "question": "Start at the cell marked by the gold dot. Follow four arrows, moving one cell each time. Which token is in the final cell?",
        "answer": answer,
        "distractors": list(distractors),
        "evidence": evidence,
        "graph": graph,
        "depth": depth,
        "guides": guides,
        "solution_trace": [grid_position(value) for value in route] + [answer],
    }


SET_SPECS = (
    (("circle", "star", "square"), ("star", "square", "diamond"), ("square", "hexagon")),
    (("circle", "star", "square", "diamond"), ("star", "square", "diamond", "hexagon"), ("diamond", "triangle")),
    (("circle", "star", "square", "diamond"), ("circle", "star", "square", "diamond", "hexagon"), ("diamond", "triangle")),
    (("circle", "star"), ("circle", "star", "square", "diamond"), ("circle", "star", "hexagon")),
)
CODE_WORDS = {0: "NOVA", 1: "LUMA", 2: "PICO", 3: "TERRA", 4: "VECTA"}


def render_set_relation(index: int, path: Path, rng: random.Random) -> dict:
    set_a, set_b, set_c = (set(value) for value in SET_SPECS[index])
    intersection = sorted(set_a & set_b)
    remaining = sorted(set(intersection) - set_c)
    count = len(remaining)
    answer = CODE_WORDS[count]
    distractors = [CODE_WORDS[value] for value in range(4) if value != count][:3]
    image, draw = base_image("Set relation", "Use the three visible sets, then use the count-to-code table.")
    panels = [(28, 122, 220, 350), (264, 122, 456, 350), (500, 122, 692, 350)]
    for panel_index, (bounds, values) in enumerate(zip(panels, (sorted(set_a), sorted(set_b), sorted(set_c)))):
        draw.rounded_rectangle(bounds, radius=14, fill=("#fee2e2", "#dbeafe", "#dcfce7")[panel_index], outline="#475569", width=2)
        draw.text(((bounds[0] + bounds[2]) // 2, bounds[1] + 24), f"SET {'ABC'[panel_index]}", anchor="mm", fill="#0f172a", font=font(18, True))
        for value_index, value in enumerate(values):
            draw.text((bounds[0] + 20, bounds[1] + 58 + value_index * 31), f"- {value}", fill="#334155", font=font(16))
    draw.rounded_rectangle((155, 378, 565, 458), radius=12, fill="#f1f5f9", outline="#475569", width=2)
    code_line = "   ".join(f"{key}={CODE_WORDS[key]}" for key in range(5))
    draw.text((360, 400), "COUNT TO CODE", anchor="mm", fill="#0f172a", font=font(16, True))
    draw.text((360, 432), code_line, anchor="mm", fill="#334155", font=font(14))
    image.save(path, format="PNG", optimize=False)

    intersection_text = ", ".join(intersection) if intersection else "empty set"
    remaining_text = ", ".join(remaining) if remaining else "empty set"
    evidence = [
        "Set A contains " + ", ".join(sorted(set_a)),
        "Set B contains " + ", ".join(sorted(set_b)),
        "Set C contains " + ", ".join(sorted(set_c)),
        f"The visible code table maps count {count} to {answer}",
    ]
    states = [
        f"The intersection of A and B is {intersection_text}",
        f"After removing members of C the set is {remaining_text}",
        f"The remaining count is {count}",
    ]
    graph, depth = chain_graph(evidence, states, f"The code word is {answer}")
    # Both A and B are required for the first state; C is required for the second.
    graph["edges"].append(["e2", "i1"])
    graph["edges"].append(["e3", "i2"])
    guides = [
        {"question": "Which shape names occur in both set A and set B?", "answer": intersection_text, "supports_next": "i1"},
        {"question": "After removing any of those names that also occur in set C, which names remain?", "answer": remaining_text, "supports_next": "i2"},
        {"question": "How many names remain after that removal?", "answer": str(count), "supports_next": "i3"},
    ]
    return {
        "question": "Find the shape names in both A and B, remove any that also occur in C, count what remains, and use the visible table. Which code word results?",
        "answer": answer,
        "distractors": distractors,
        "evidence": evidence,
        "graph": graph,
        "depth": depth,
        "guides": guides,
        "solution_trace": ["A intersection B", intersection_text, remaining_text, str(count), answer],
    }


RENDERERS = (
    ("directed_path", render_directed_path),
    ("sequential_transform", render_sequential_transform),
    ("grid_trace", render_grid_trace),
    ("set_relation", render_set_relation),
)


def generation_spec() -> dict:
    return {
        "generator_id": GENERATOR_ID,
        "seed": SEED,
        "canvas": CANVAS,
        "families": [family for family, _ in RENDERERS],
        "items_per_family": 4,
        "correct_label_schedule": list("ABCD") * 4,
        "path_specs": PATH_SPECS,
        "transform_specs": TRANSFORM_SPECS,
        "grid_specs": GRID_SPECS,
        "set_specs": SET_SPECS,
        "code_words": CODE_WORDS,
        "font_sha256": {"regular": sha256_file(FONT_PATH), "bold": sha256_file(BOLD_FONT_PATH)},
        "generator_code_sha256": sha256_file(Path(__file__).resolve()),
        "contract_code_sha256": sha256_file(CODE_ROOT / "evaluation" / "g_pc0.py"),
        "pillow_version": pillow_version,
    }


def generate_bundle(output_dir: Path, image_prefix: str, asset_root: Path | None = None) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not image_prefix or Path(image_prefix).is_absolute() or ".." in Path(image_prefix).parts:
        raise ValueError("image-prefix must be a non-empty relative path without '..'")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)
    rng = random.Random(SEED)
    spec = generation_spec()
    spec_hash = sha256_bytes(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    rows = []
    for family_index, (family, renderer) in enumerate(RENDERERS):
        for item_index in range(4):
            global_index = family_index * 4 + item_index
            image_name = f"gpc0-{global_index + 1:02d}-{family}.png"
            image_path = images_dir / image_name
            rendered = renderer(item_index, image_path, rng)
            correct_label = "ABCD"[global_index % 4]
            options = place_options(rendered["answer"], rendered["distractors"], correct_label)
            rows.append(common_record(
                global_index=global_index,
                family=family,
                image_path=str(Path(image_prefix) / image_name),
                image_checksum=sha256_file(image_path),
                question=rendered["question"],
                options=options,
                correct_label=correct_label,
                evidence=rendered["evidence"],
                graph=rendered["graph"],
                depth=rendered["depth"],
                guides=rendered["guides"],
                solution_trace=rendered["solution_trace"],
                spec_hash=spec_hash,
            ))
    errors = validate_g_pc0_manifest(rows, expected_n=16)
    root = asset_root if asset_root is not None else Path(__file__).resolve().parents[4]
    errors.extend(validate_g_pc0_assets(rows, root))
    direct_errors = []
    for row in rows:
        generated = images_dir / Path(row["image_original"]).name
        if sha256_file(generated) != row["image_checksum"]:
            direct_errors.append(f"{row['sample_id']}: direct generated checksum mismatch")
    if direct_errors:
        errors.extend(direct_errors)
    if errors:
        raise ValueError("generated G-PC0 bundle failed validation: " + "; ".join(sorted(set(errors))))

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("x", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    spec_path = output_dir / "generation-spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "manifest_profile": PROFILE,
        "generator_id": GENERATOR_ID,
        "generator_seed": SEED,
        "records": len(rows),
        "families": {family: 4 for family, _ in RENDERERS},
        "answer_labels": {label: 4 for label in "ABCD"},
        "generation_spec_sha256": spec_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "image_sha256": {path.name: sha256_file(path) for path in sorted(images_dir.glob("*.png"))},
        "human_verified_records": 0,
        "procedural_gold_records": 16,
        "natural_g0_records": 0,
        "training_records": 0,
        "gpu_used": False,
        "validation_errors": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-prefix", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        raise SystemExit(f"G-PC0 v1 seed is frozen at {SEED}")
    summary = generate_bundle(args.output_dir, args.image_prefix)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
