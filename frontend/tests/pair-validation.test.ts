import { describe, expect, it } from "vitest";
import { validatePair } from "../src/api/bundles";

/**
 * Pair designation is validated client-side so a mismatch blocks the Prepare
 * control instead of costing a round trip and returning a 409 PAIR_INVALID.
 * These are the rules the button depends on.
 */
describe("validatePair", () => {
  it("accepts a crossmodal pair with one optical and one SAR scene", () => {
    const result = validatePair("crossmodal", [
      { role: "optical", acquired_at: null },
      { role: "sar", acquired_at: null },
    ]);
    expect(result.valid).toBe(true);
    expect(result.reason).toBeNull();
  });

  it("rejects a crossmodal pair missing the SAR role", () => {
    const result = validatePair("crossmodal", [
      { role: "optical", acquired_at: null },
      { role: "optical", acquired_at: null },
    ]);
    expect(result.valid).toBe(false);
    expect(result.reason).toContain("SAR");
  });

  it("requires acquisition dates on both halves of a bi-temporal pair", () => {
    const withoutDates = validatePair("bitemporal", [
      { role: "t1", acquired_at: null },
      { role: "t2", acquired_at: null },
    ]);
    expect(withoutDates.valid).toBe(false);
    expect(withoutDates.reason).toContain("acquisition date");

    const withDates = validatePair("bitemporal", [
      { role: "t1", acquired_at: "2024-03-11T05:20:00Z" },
      { role: "t2", acquired_at: "2024-03-18T05:20:00Z" },
    ]);
    expect(withDates.valid).toBe(true);
  });

  it("rejects a pair with the wrong number of scenes", () => {
    expect(
      validatePair("crossmodal", [{ role: "optical", acquired_at: null }]).valid,
    ).toBe(false);
    expect(
      validatePair("single", [
        { role: "optical", acquired_at: null },
        { role: "optical", acquired_at: null },
      ]).valid,
    ).toBe(false);
  });

  it("reports the roles a pair type requires", () => {
    expect(validatePair("bitemporal", []).requiredRoles).toEqual(["t1", "t2"]);
    expect(validatePair("crossmodal", []).requiredRoles).toEqual([
      "optical",
      "sar",
    ]);
  });
});
