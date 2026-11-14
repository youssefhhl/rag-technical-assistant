// Sample conversation shown when the page is opened with `?demo`.
// Used to produce the README screenshot without running the backend.

import type { Citation } from "./api";

export interface DemoMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

export const DEMO_MESSAGES: DemoMessage[] = [
  {
    role: "user",
    content: "How do I declare a path parameter in FastAPI?",
  },
  {
    role: "assistant",
    content:
      "Declare the parameter in the path string with the same syntax as Python format strings, " +
      "and add an argument with the same name to your function:\n\n" +
      '```python\n@app.get("/items/{item_id}")\nasync def read_item(item_id: int):\n    return {"item_id": item_id}\n```\n\n' +
      "The type annotation (`int`) makes FastAPI convert the value and validate it. " +
      "A request to `/items/foo` returns a clear validation error instead of reaching your code.",
    citations: [
      {
        title: "Path Parameters",
        source_url: "https://fastapi.tiangolo.com/tutorial/path-params/",
        chunk_index: 0,
      },
      {
        title: "Path Parameters > Path parameters with types",
        source_url: "https://fastapi.tiangolo.com/tutorial/path-params/",
        chunk_index: 1,
      },
      {
        title: "Path Parameters > Data validation",
        source_url: "https://fastapi.tiangolo.com/tutorial/path-params/",
        chunk_index: 3,
      },
    ],
  },
];
