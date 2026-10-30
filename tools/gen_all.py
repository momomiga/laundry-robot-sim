#!/usr/bin/env python3
"""Regenerate every model from tools/params.py."""
import gen_room, gen_washer, gen_robot

for m in (gen_room, gen_washer, gen_robot):
    m.OUT.parent.mkdir(parents=True, exist_ok=True)
    m.OUT.write_text(m.build())
    print("wrote", m.OUT)
