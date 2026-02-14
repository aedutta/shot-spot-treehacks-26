#!/bin/bash
if [ -d ".venv/Scripts" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
