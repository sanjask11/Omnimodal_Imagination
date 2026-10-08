#!/usr/bin/env python3
"""Pipeline 2 (dry speech): GPT writes an AUDIO-IMAGINATION preamble.
The original SpatialSceneQA question (fixed task + answer format) is appended unchanged."""
import json, argparse, time
from collections import defaultdict
from openai import OpenAI

# Original SpatialSceneQA-style questions: fixed task + answer format (never written by GPT)
ORIG = {
 "A_doa":       "Based on the audio, output the precise azimuth and elevation of the sound source in degrees.",
 "B_doa":       "Where is the {role} voice coming from? Output the azimuth and elevation in degrees.",
 "dir_mcq":     "Which direction is the sound coming from: Left, Center, or Right?",
 "E_matching":  "Which speaker does the {role} voice originate from: Left, Center, or Right?",
 "C_grounding": "Identify the 3D bounding box of the sound-emitting speaker in this scene.",
}

# Goal shown to GPT only as context
GOALS = {
 "A_doa":       "estimate the azimuth and elevation of the sound source",
 "B_doa":       "estimate the azimuth and elevation of the {role} voice",
 "dir_mcq":     "decide whether the sound comes from the Left, Center, or Right",
 "E_matching":  "decide which loudspeaker is producing the {role} voice",
 "C_grounding": "give the 3D bounding box of the sound-emitting speaker",
}

SYS = (
 "You write the IMAGINATION PREAMBLE for a spatial audio-visual question (Pipeline 2: DRY SPEECH). "
 "The model under test receives an RGB image, a depth map, and a DRY speech clip: a clean voice with "
 "no echo, no reverb, no direction and no distance cues. "
 "IMPORTANT: you CANNOT see the scene. Do NOT state any facts about it - no distances, positions, sides, "
 "materials, room size, or which loudspeaker is where. Write instructions, not observations. "
 "In 2-3 sentences, tell the model to: find each visible loudspeaker in the image and judge how far it is "
 "using the depth map; imagine, in the audio domain, how the dry voice would sound if it played from each "
 "one (how loud, how much echo, given the room size and surfaces it can see); and use that imagined sound "
 "to answer. Use plain language. Do NOT write the final question or any answer options - they will be "
 "appended. Do not hint at the answer."
)

def make_prompt(it):
    n = "one visible loudspeaker" if it["config"] == "single_source" else "two visible loudspeakers"
    goal = GOALS[it["task"]].format(role=it.get("target_role") or "")
    return (f"Scene type: {n}. Input to the model: DRY speech (no room acoustics).\n"
            f"The question that follows will ask the model to {goal}.\n"
            f"Write the imagination preamble only.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerkey", default="poc_qa.json")
    ap.add_argument("--out", default="questions/questions_p2.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--per_task", type=int, default=2)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    client = OpenAI()

    items = json.load(open(a.answerkey))["items"]
    byt = defaultdict(list)
    for it in items:
        if a.tasks and it["task"] not in a.tasks: continue
        byt[it["task"]].append(it)
    selected = [x for t, l in byt.items() for x in l[:a.per_task]]
    print(f"Selected {len(selected)} items across {len(byt)} tasks (seed={a.seed}, temp={a.temperature})")

    out = []
    for i, it in enumerate(selected):
        role = it.get("target_role") or ""
        orig_q = ORIG[it["task"]].format(role=role)
        try:
            r = client.chat.completions.create(
                model=a.model, seed=a.seed, temperature=a.temperature, max_tokens=200,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": make_prompt(it)}])
            pre = r.choices[0].message.content.strip()
        except Exception as e:
            pre = None; print("ERR", i, e)
        out.append({**{k: it.get(k) for k in ("scene_id","sample_id","task","config","target_role","answer")},
                    "pipeline": "p2_dry", "seed": a.seed, "temperature": a.temperature,
                    "preamble": pre, "original_question": orig_q,
                    "question": (pre + "\n\n" + orig_q) if pre else None})
        print(f"[{i+1}/{len(selected)}] {it['task']}")
        time.sleep(0.3)

    json.dump({"n": len(out), "pipeline": "p2_dry", "seed": a.seed, "temperature": a.temperature,
               "model": a.model, "items": out}, open(a.out, "w"), indent=2)
    print(f"Wrote {len(out)} -> {a.out}")

if __name__ == "__main__":
    main()