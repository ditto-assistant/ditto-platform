# Moved to ditto-subnet

Active Platform development and deployment now live in
[`ditto-assistant/ditto-subnet/apps/platform`](https://github.com/ditto-assistant/ditto-subnet/tree/main/apps/platform).

This cutover removes the old deploy, migration-order, anti-copy refresh, and
cross-repository Backroom contract workflows. Their monorepo replacements use
the same checked-out release as Backroom and the screening protocol, so no
contract-dispatch or version-bump pull request is required.

Merge this cutover only after the destination monorepo and infra PR stacks are
ready and the protected `dev`/`prod` environments have their monorepo WIF
bindings. This repository remains readable for history.
