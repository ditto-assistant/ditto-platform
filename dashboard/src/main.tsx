import { render } from "solid-js/web";

import App from "./App";
import "./styles/index.css";

const root = document.getElementById("root");

if (!root) throw new Error("Dashboard root element is missing");

render(() => <App />, root);
