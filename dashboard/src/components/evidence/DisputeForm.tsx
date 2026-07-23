import { Show, createSignal } from "solid-js";
import type { JSX } from "solid-js";

import { postJSON } from "../../lib/api";

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}

async function signingPayload(agentId: string, message: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(message));
  const hash = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  return `ditto-dispute-v1:${agentId}:${hash}`;
}

export function DisputeForm(props: { agentId: string; onSubmitted: () => void }): JSX.Element {
  const [message, setMessage] = createSignal("");
  const [wallet, setWallet] = createSignal("");
  const [hotkey, setHotkey] = createSignal("");
  const [signature, setSignature] = createSignal("");
  const [command, setCommand] = createSignal("");
  const [state, setState] = createSignal("");
  const validMessage = () => message().trim().length >= 20 && message().trim().length <= 1000;
  const validSignature = () => /^[0-9a-fA-F]{128}$/.test(signature().trim());
  const updateCommand = async () => {
    if (!validMessage() || !wallet().trim() || !hotkey().trim()) {
      setCommand("");
      return;
    }
    const payload = await signingPayload(props.agentId, message().trim());
    setCommand(
      `btcli wallet sign --wallet-name ${shellQuote(wallet().trim())} --wallet-hotkey ${shellQuote(hotkey().trim())} --use-hotkey --message ${shellQuote(payload)} --json-output`,
    );
  };
  const submit = async (event: SubmitEvent) => {
    event.preventDefault();
    if (!validMessage() || !validSignature()) return;
    setState("Submitting dispute…");
    try {
      await postJSON(`/public/agent/${encodeURIComponent(props.agentId)}/dispute`, {
        message: message().trim(),
        signature: signature().trim(),
      });
      setState("Dispute submitted.");
      props.onSubmitted();
    } catch (error) {
      setState(error instanceof Error ? error.message : "The dispute could not be submitted.");
    }
  };
  return (
    <section class="pipeline-section screening-dispute">
      <div class="pipeline-section-heading">
        <h4>Dispute screening decision</h4>
      </div>
      <p>
        You may submit one private, hotkey-signed dispute for this rejected submission. Once
        submitted, it cannot be edited or replaced.
      </p>
      <form class="screening-dispute-form" onSubmit={submit}>
        <label>
          Your dispute
          <textarea
            minlength="20"
            maxlength="1000"
            required
            value={message()}
            onInput={(event) => {
              setMessage(event.currentTarget.value);
              void updateCommand();
            }}
          />
          <span>{message().length} / 1000</span>
        </label>
        <div class="screening-dispute-wallets">
          <label>
            Wallet name
            <input
              value={wallet()}
              onInput={(event) => {
                setWallet(event.currentTarget.value);
                void updateCommand();
              }}
              placeholder="default"
            />
          </label>
          <label>
            Hotkey name
            <input
              value={hotkey()}
              onInput={(event) => {
                setHotkey(event.currentTarget.value);
                void updateCommand();
              }}
              placeholder="miner"
            />
          </label>
        </div>
        <Show
          when={command()}
          fallback={
            <p>
              Enter a valid dispute, wallet name, and hotkey name to generate the local signing
              command.
            </p>
          }
        >
          <label>
            Ready-to-run btcli command
            <pre>
              <code>{command()}</code>
            </pre>
          </label>
        </Show>
        <label>
          Hotkey signature
          <input
            value={signature()}
            onInput={(event) => setSignature(event.currentTarget.value)}
            maxlength="128"
            pattern="[0-9a-fA-F]{128}"
            placeholder="128-character hexadecimal signature"
            required
          />
        </label>
        <button class="btn" type="submit" disabled={!validMessage() || !validSignature()}>
          Submit final dispute
        </button>
        <p role="status" aria-live="polite">
          {state()}
        </p>
      </form>
    </section>
  );
}
