# make release   build target/release/gibbs
# make tui       build, then open the dashboard
#                extra flags: make tui ARGS="--model THUDM/glm-edge-1.5b-chat --backend metal --port 8081"

ARGS ?=

.PHONY: release tui

release:
	cargo build --release

tui: release
	./target/release/gibbs --tui $(ARGS)
