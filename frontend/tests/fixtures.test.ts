import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import Ajv2020 from "ajv/dist/2020";

/**
 * The alarm the plan asks for (§9.4).
 *
 * Every trace fixture is validated against `configs/trace_schema.json`. If
 * the backend changes the schema without a version bump, these fixtures fail
 * first — which is exactly where we want to find out, rather than in a demo.
 */

const REPO_ROOT = join(import.meta.dirname, "..", "..");
const FIXTURE_DIR = join(import.meta.dirname, "..", "mocks", "fixtures");
const SCHEMA_PATH = join(REPO_ROOT, "configs", "trace_schema.json");

const schema = JSON.parse(readFileSync(SCHEMA_PATH, "utf8"));

const traceFixtures = readdirSync(FIXTURE_DIR).filter((name) =>
  name.startsWith("trace_"),
);

describe("trace fixtures", () => {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  const validate = ajv.compile(schema);

  it("ships at least the three fixtures the plan names", () => {
    expect(traceFixtures).toEqual(
      expect.arrayContaining([
        "trace_crossmodal_success.json",
        "trace_refusal_missing_input.json",
        "trace_param_rejected.json",
      ]),
    );
  });

  it.each(traceFixtures)("%s matches trace_schema.json v2", (name) => {
    const fixture = JSON.parse(
      readFileSync(join(FIXTURE_DIR, name), "utf8"),
    );
    const valid = validate(fixture);
    if (!valid) {
      throw new Error(
        `${name} does not match the frozen trace schema:\n${ajv.errorsText(
          validate.errors,
          { separator: "\n" },
        )}`,
      );
    }
    expect(valid).toBe(true);
  });

  it("declares schema_version 2 everywhere", () => {
    for (const name of traceFixtures) {
      const fixture = JSON.parse(
        readFileSync(join(FIXTURE_DIR, name), "utf8"),
      );
      expect(fixture.schema_version, name).toBe(2);
    }
  });
});
