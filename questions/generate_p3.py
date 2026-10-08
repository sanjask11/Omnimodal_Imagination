#!/usr/bin/env python3
"""Pipeline 3 (speech WITH room acoustics): GPT writes an AUDIO-IMAGINATION preamble.
The original SpatialSceneQA question (fixed task + answer format) is appended unchanged."""
import json, argparse, time
from collections import defaultdict
from openai import OpenAI

# Original SpatialSceneQA-style questions (never written by GPT)
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
 "You write the IMAGINATION PREAMBLE for a spatial audio-visual question (Pipeline 3: SPEECH WITH ROOM "
 "ACOUSTICS). The model receives an RGB image, a depth map, and a single-channel recording of the voice as "
 "it actually sounds in this room: it includes the room's echo/reverberation and the loudness drop from "
 "distance, but no left/right direction. "
 "IMPORTANT: you CANNOT see the scene or hear the audio. Never state facts about either - no distances, "
 "positions, sides, materials, room size, how loud or echoey the voice is, or which loudspeaker is where. "
 "Write instructions, not observations. "
 "In 2-4 sentences, tell the model to: (1) listen to how loud the voice is and how much echo it has, and "
 "imagine roughly how far away the source must be; (2) find each visible loudspeaker and judge its "
 "distance using the depth map; (3) if there are two loudspeakers, imagine how the voice would sound from "
 "each one and pick the one that best matches what it hears; if there is only one, imagine how the voice "
 "would sound from it and check that this fits what it hears; (4) then use where that loudspeaker sits in "
 "the image to answer. If the question targets a specific voice (male or female), tell it to focus on that "
 "voice in the recording. "
 "Call the input 'the voice recording' (no jargon). Use plain language. Do NOT write the final question or "
 "any answer options - they will be appended. Do not hint at the answer."
)

def make_prompt(it):
    single = it["config"] == "single_source"
    n = "one visible loudspeaker" if single else "two visible loudspeakers"
    voices = "one voice" if single else "two overlapping voices (one male, one female)"
    role = it.get("target_role") or ""
    focus = f" The question targets the {role} voice." if role else ""
    goal = GOALS[it["task"]].format(role=role)
    return (f"Scene type: {n}. Audio content: {voices}.{focus}\n"
            f"The question that follows will ask the model to {goal}.\n"
            f"Write the imagination preamble only.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerkey", default="poc_qa.json")
    ap.add_argument("--out", default="questions/questions_p3.json")
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
    print(f"[p3] {len(selected)} items across {len(byt)} tasks (seed={a.seed}, temp={a.temperature})")

    out = []
    for i, it in enumerate(selected):
        role = it.get("target_role") or ""
        orig_q = ORIG[it["task"]].format(role=role)
        try:
            r = client.chat.completions.create(
                model=a.model, seed=a.seed, temperature=a.temperature, max_tokens=220,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": make_prompt(it)}])
            pre = r.choices[0].message.content.strip()
        except Exception as e:
            pre = None; print("ERR", i, e)
        out.append({**{k: it.get(k) for k in ("scene_id","sample_id","task","config","target_role","answer")},
                    "pipeline": "p3_room", "seed": a.seed, "temperature": a.temperature,
                    "preamble": pre, "original_question": orig_q,
                    "question": (pre + "\n\n" + orig_q) if pre else None})
        print(f"[{i+1}/{len(selected)}] {it['task']}")
        time.sleep(0.3)

    json.dump({"n": len(out), "pipeline": "p3_room", "seed": a.seed, "temperature": a.temperature,
               "model": a.model, "items": out}, open(a.out, "w"), indent=2)
    print(f"Wrote {len(out)} -> {a.out}")

if __name__ == "__main__":
    main()