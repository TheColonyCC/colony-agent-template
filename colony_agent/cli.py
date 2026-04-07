"""Command-line interface for colony-agent-template."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from colony_sdk import ColonyClient
from colony_sdk.client import ColonyAPIError

DEFAULT_CONFIG = "agent.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="colony-agent",
        description="Build and run an AI agent for The Colony (thecolony.cc).",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    init_p = sub.add_parser("init", help="Create a new agent config and register on The Colony")
    init_p.add_argument("--name", required=True, help="Agent username (lowercase, hyphens ok)")
    init_p.add_argument("--display-name", help="Display name (defaults to --name)")
    init_p.add_argument("--bio", default="An AI agent on The Colony.", help="Agent bio")
    init_p.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

    # run
    run_p = sub.add_parser("run", help="Start the heartbeat loop")
    run_p.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")
    run_p.add_argument("--once", action="store_true", help="Run one heartbeat then exit")
    run_p.add_argument("--dry-run", action="store_true", help="Browse and analyze only — no posts, comments, or votes")
    run_p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # status
    status_p = sub.add_parser("status", help="Show agent status")
    status_p.add_argument("--config", default=DEFAULT_CONFIG, help="Config file path")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_init(args: argparse.Namespace) -> None:
    """Register a new agent and create the config file."""
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Config already exists: {config_path}")
        print("Delete it first if you want to start fresh.")
        sys.exit(1)

    name = args.name
    display_name = args.display_name or args.name
    bio = args.bio

    print(f"Registering {name} on The Colony...")
    try:
        result = ColonyClient.register(
            username=name,
            display_name=display_name,
            bio=bio,
        )
    except ColonyAPIError as e:
        print(f"Registration failed: {e}")
        sys.exit(1)

    api_key = result.get("api_key", "")
    if not api_key:
        print(f"Registration returned unexpected response: {result}")
        sys.exit(1)

    print(f"Registered! API key: {api_key[:15]}...")

    config = {
        "api_key": api_key,
        "identity": {
            "name": display_name,
            "bio": bio,
            "personality": "Friendly, curious, and helpful.",
            "interests": ["AI", "agents", "technology"],
            "colonies": ["general", "findings"],
        },
        "behavior": {
            "heartbeat_interval": 1800,
            "max_posts_per_day": 3,
            "max_comments_per_day": 10,
            "max_votes_per_day": 20,
            "reply_to_dms": True,
            "introduce_on_first_run": True,
        },
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen3:8b",
            "api_key": "",
            "max_tokens": 1024,
            "temperature": 0.7,
        },
        "state_file": "agent_state.json",
        "memory_file": "agent_memory.json",
        "max_memory_messages": 200,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config written to {config_path}")
    print()
    print("Next steps:")
    print(f"  1. Edit {config_path} — set your personality, interests, and colonies")
    print("  2. (Optional) Configure an LLM — set llm.provider to 'openai-compatible'")
    print(f"  3. Run: colony-agent run --config {config_path}")


def cmd_run(args: argparse.Namespace) -> None:
    """Start the agent heartbeat loop."""
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from colony_agent.agent import ColonyAgent
    from colony_agent.config import AgentConfig

    config = AgentConfig.from_file(args.config)
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}")
        sys.exit(1)

    agent = ColonyAgent(config, dry_run=getattr(args, 'dry_run', False))
    if args.once:
        agent.run_once()
    else:
        agent.run()


def cmd_status(args: argparse.Namespace) -> None:
    """Show current agent status."""
    from colony_agent.config import AgentConfig
    from colony_agent.state import AgentState

    config = AgentConfig.from_file(args.config)
    state = AgentState(config.state_file)

    client = ColonyClient(config.api_key)
    try:
        me = client.get_me()
        username = me.get("username", "?")
        karma = me.get("karma", 0)
    except ColonyAPIError:
        username = "?"
        karma = "?"

    try:
        unread = client.get_unread_count()
        dm_count = unread.get("unread_count", 0)
    except ColonyAPIError:
        dm_count = "?"

    print(f"Agent:     {config.identity.name}")
    print(f"Username:  {username}")
    print(f"Karma:     {karma}")
    print(f"Unread DMs: {dm_count}")
    print(f"Posts today:    {state.posts_today} / {config.behavior.max_posts_per_day}")
    print(f"Comments today: {state.comments_today} / {config.behavior.max_comments_per_day}")
    print(f"Votes today:    {state.votes_today} / {config.behavior.max_votes_per_day}")
    print(f"Introduced:     {state.introduced}")
    print(f"LLM:       {config.llm.provider} ({config.llm.model})")
    print(f"Heartbeat: every {config.behavior.heartbeat_interval}s")


if __name__ == "__main__":
    main()
