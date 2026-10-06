`timescale 1ns/1ps
module testbench;

reg clk;
reg reset;
reg [7:0] d;

wire [7:0] q;

top_module uut (
    .clk(clk),
    .reset(reset),
    .d(d),
    .q(q)
);

integer cycle;
integer MAX_CYCLES;
reg [2:0] exp_count;
reg [7:0] exp_dff;
reg [7:0] expected_q;

// Additional helpers
integer phase; // 0 = initial smoke, 1 = directed tests running, 2 = finished
integer seed;
integer rnd;
integer i;

reg [2:0] prev_count; // previous count to detect transitions

initial begin
    $dumpfile("wave_top_module.vcd");
    $dumpvars(0, testbench.uut);
end

initial begin
    // deterministic seed for reproducible randomized stimulus
    seed = 32'hDEADBEEF;
    // initialize RNG (some simulators seed and return a value)
    rnd = $urandom(seed);

    clk = 0;
    reset = 1;
    d = 8'hA5;
    exp_count = 3'b000;
    exp_dff = 8'h00;
    cycle = 0;
    MAX_CYCLES = 1000;
    phase = 0;
    prev_count = 3'b000;

    #12;
    reset = 0;

    #37;
    reset = 1;      // asynchronous assert of counter reset (not aligned to posedge)
    #7;
    reset = 0;      // release

    #50;
    // another asynchronous reset pulse
    reset = 1;
    #3;
    reset = 0;

    // let simulation run, additional directed tests will start after the baseline smoke test
end

always #5 clk = ~clk; // 10ns period

// Model of the counter (synchronous on posedge clk, asynchronous on posedge reset)
always @(posedge clk or posedge reset) begin
    if (reset) begin
        exp_count <= 3'b000;
    end else begin
        if (exp_count == 3'b111)
            exp_count <= 3'b000;
        else
            exp_count <= exp_count + 1;
    end
end

// Model of the DFF with asynchronous reset tied to MSB of counter (exp_count[2])
always @(posedge clk or posedge exp_count[2]) begin
    if (exp_count[2])
        exp_dff <= 8'h00;
    else
        exp_dff <= d;
end

// Drive data on negedge so it is stable before next posedge
always @(negedge clk) begin
    // vary data: mix of increment and rotated pattern
    if (cycle % 4 == 0)
        d <= d + 8'h1;
    else if (cycle % 4 == 1)
        d <= {d[6:0], d[7]}; // rotate left
    else if (cycle % 4 == 2)
        d <= d ^ 8'hFF;
    else
        d <= d - 8'h3;
end

// Baseline checks (keeps original checks intact)
// We will not finish at cycle 30; instead we mark completion of the baseline smoke test and continue to directed tests.
always @(posedge clk) begin
    #1;
    cycle = cycle + 1;
    if (cycle > MAX_CYCLES) begin
        $fatal(1, "Error: Timeout reached without completing checks (cycle=%0d)", cycle);
    end

    // read DUT internals (hierarchical) to verify internal count and stored dff output
    if (uut.count !== exp_count) begin
        $fatal(0, "Error: Counter mismatch at cycle %0d: DUT.count=%b expected=%b", cycle, uut.count, exp_count);
    end

    if (uut.dff_out !== exp_dff) begin
        $fatal(0, "Error: DFF output mismatch at cycle %0d: DUT.dff_out=0x%0h expected=0x%0h", cycle, uut.dff_out, exp_dff);
    end

    // expected top q: equal to exp_dff only when count == 3'b111 (original test expectation)
    if (exp_count == 3'b111)
        expected_q = exp_dff;
    else
        expected_q = 8'h00;

    if (q !== expected_q) begin
        $fatal(0, "Error: Top q mismatch at cycle %0d: DUT.q=0x%0h expected=0x%0h (count=%b)", cycle, q, expected_q, exp_count);
    end

    // end of baseline smoke test: keep the test passing, but do not finish here; move to directed tests
    if (cycle == 30 && phase == 0) begin
        $display("Baseline smoke test passed to cycle %0d. Starting directed stress tests to exercise asynchronous behaviors.", cycle);
        phase = 1;
    end
end

// New directed/randomized tests and invariant checks started after baseline
initial begin
    // Wait until baseline smoke completes
    wait (phase == 1);

    // Directed test #1:
    // Target: exercise asynchronous DFF reset triggered by count[2] transitions and misaligned top.reset interactions.
    // Rationale: try more timing combinations to exercise edge-cases and toggles in counter/dff paths.
    $display("Directed test 1: asynchronous reset and misaligned pulses.");
    // Align some actions to clock edges to create varied timing
    @(posedge clk); @(posedge clk);
    // Drive data that has low 3 bits set (attempt to produce non-zero q if possible)
    d <= 8'hF7; // lower 3 bits are 111
    // Wait several cycles while toggling top reset in unusual alignments
    for (i = 0; i < 8; i = i + 1) begin
        @(negedge clk);
        // misaligned asynchronous reset pulse (short)
        reset = 1;
        #3;
        reset = 0;
        @(posedge clk);
    end

    // Directed test #1b:
    // Target: pulse top.reset around posedge to create different ordering of counter update and DFF async reset.
    // Rationale: exercise combination of posedge counter update and async to DFF.
    $display("Directed test 1b: pulses around posedge to exercise ordering.");
    // Wait for a moment to hit multiple phases of counter
    @(posedge clk);
    #1;
    reset = 1; // assert asynchronous reset just after posedge
    #4;
    reset = 0;
    // do this a few times at different phases
    @(posedge clk); @(posedge clk);
    #2; reset = 1; #1; reset = 0;
    @(posedge clk);

    // Directed test #2:
    // Target: assert invariants and run more reproducible random stimuli (documenting the 'q never toggles upper bits' behavior).
    $display("Directed test 2: invariants and more stimulus (seed=%0h).", seed);

    // Provide reproducible pseudo-random stimuli for a while
    for (i = 0; i < 120; i = i + 1) begin
        // deterministic randomness: advance RNG using seeded $urandom
        rnd = $urandom(seed ^ i);
        // set d to various patterns including ones with lower 3 bits set/unset
        d <= rnd[7:0];

        // occasionally toggle top reset asynchronously to stress counter/dff interaction
        if ((rnd & 7) == 0) begin
            // pulse reset around a negedge to create an asynchronous boundary
            @(negedge clk);
            reset = 1;
            #2;
            reset = 0;
            // align back to posedge to let counter/dff settle
            @(posedge clk);
        end else if ((rnd & 7) == 1) begin
            // pulse reset short around posedge to try different ordering
            @(posedge clk);
            #1; reset = 1; #1; reset = 0;
        end else begin
            // just wait one clock
            @(posedge clk);
        end
    end

    // Directed test #3:
    // Target: try to create the corner-case where count==111 and dff_out could be non-zero (verify it's impossible by checking invariants).
    // Rationale: exercise timing windows; this test documents that asynchronously reset prevents any q non-zero at that time.
    $display("Directed test 3: trying to corner-case dff_out vs count==111.");
    // Drive d to 0xFF for several cycles so when dff can sample it, it will have low bits set.
    for (i = 0; i < 12; i = i + 1) begin
        d <= 8'hFF;
        @(posedge clk);
    end

    // Additionally try tight timing where we toggle d close to posedge and induce asynchronous pulses to see if any transient occurs
    // (this is deterministic, still seed-based)
    for (i = 0; i < 20; i = i + 1) begin
        // change d just before posedge
        @(negedge clk);
        d <= (i % 2) ? 8'h0F : 8'hF0;
        // sometimes assert a very short reset after negedge to create unusual timing
        if ((i & 3) == 0) begin
            #1; reset = 1; #1; reset = 0;
        end
        @(posedge clk);
    end

    // More directed pulses to exercise counter wrap and transitions (3->4, 6->7, 7->0)
    $display("Directed test 4: transition and wrap checks.");
    // Wait until we see a specific count and then apply targeted pulses
    // Wait for an occurrence of count == 3 (so next posedge will go to 4 where MSB rises)
    repeat (200) begin
        @(posedge clk);
        if (uut.count == 3) begin
            // on the cycle where count==3, manipulate timing to try to exercise asynchronous reaction
            @(negedge clk);
            d <= 8'hAA;
            // produce an asynchronous activity (short pulses) to stress edge ordering
            reset = 1;
            #2;
            reset = 0;
            @(posedge clk); // now observe the reaction as count goes to 4
            break;
        end
    end

    // Allow a few cycles to settle and then finish directed tests
    for (i = 0; i < 40; i = i + 1) begin
        @(posedge clk);
    end

    // Mark directed tests complete and allow final invariant checks to run
    phase = 2;
    $display("Directed tests complete at cycle %0d. Proceeding to final checks.", cycle);
end

// Additional invariant checks executed on each posedge once directed tests are active.
// These checks codify observed behavior and help document coverage-exercised conditions.
// - q upper bits should remain zero (q[7:3] == 0) due to mask width behavior
// - when count[2] rises (asynchronous reset to dff) dff_out must be zero
// - record transitions to check reset ordering and counter wrap
always @(posedge clk) begin
    #1;
    if (phase >= 1) begin
        // Invariant 1: upper bits of q (7:3) are always zero for the DUT as written
        if (q[7:3] !== 5'b00000) begin
            $fatal(0, "Error: Unexpected q upper bits non-zero at cycle %0d: q=0x%0h count=%b dff_out=0x%0h",
                   cycle, q, uut.count, uut.dff_out);
        end

        // Invariant 2: when count==111 the AND mask enables low bits but dff is asynchronously reset by count[2] -> q should be zero
        if (uut.count === 3'b111) begin
            if (q !== 8'h00) begin
                $fatal(0, "Error: Unexpected non-zero q when count==111 at cycle %0d: q=0x%0h dff_out=0x%0h",
                       cycle, q, uut.dff_out);
            end
        end

        // Invariant 3: whenever count[2] rises (prev_count[2]==0 && uut.count[2]==1) the dff_out must be reset asynchronously
        if (prev_count[2] === 1'b0 && uut.count[2] === 1'b1) begin
            // This asserts that asynchronous reset triggers and clears the DFF immediately (observable on that cycle).
            if (uut.dff_out !== 8'h00) begin
                $fatal(0, "Error: DFF not asynchronously reset when count[2] rose at cycle %0d: count from %b->%b dff_out=0x%0h",
                       cycle, prev_count, uut.count, uut.dff_out);
            end
        end

        // Invariant 4: Counter wrap behavior check: when count transitions from 7->0 ensure exp model match (sanity)
        if (prev_count === 3'b111 && uut.count !== 3'b000) begin
            $fatal(0, "Error: Counter did not wrap correctly at cycle %0d: prev_count=111 current=%b", cycle, uut.count);
        end

        // update prev_count for next cycle
        prev_count = uut.count;
    end
end

// Final termination: when directed tests done and we have run a few more cycles, finish cleanly.
always @(posedge clk) begin
    if (phase == 2 && cycle >= 250) begin
        $display("All checks passed through directed tests up to cycle %0d, finishing simulation.", cycle);
        $finish;
    end
end

endmodule