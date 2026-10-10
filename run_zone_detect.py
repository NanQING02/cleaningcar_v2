if __name__ == '__main__':
    from pathlib import Path
    from cleaningcar.rga3_guard import GuardError, bootstrap

    try:
        bootstrap(Path(__file__).resolve().parent)
    except (GuardError, OSError) as exc:
        raise SystemExit(f'[rga3-guard] 拒绝启动：{exc}') from exc

    from cleaningcar.cli import main
    main()
