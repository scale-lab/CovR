// dff_reset.v
module dff_reset (
    input clk,
    input reset,
    input [7:0] d,
    output [7:0] q
);

reg [7:0] q_reg;

always @(posedge clk, posedge reset) begin
    if (reset) begin
        q_reg <= 8'b0;
    end
    else begin
        q_reg <= d;
    end
end

assign q = q_reg;

endmodule

// counter.v
module counter (
    input clk,
    input reset,
    output reg [2:0] count
);

always @(posedge clk, posedge reset) begin
    if (reset) begin
        count <= 3'b0;
    end
    else begin
        if (count == 3'b111) begin
            count <= 3'b0;
        end
        else begin
            count <= count + 1;
        end
    end
end

endmodule

// and_module.v
module and_module (
    input [7:0] d,
    input [2:0] count,
    output [7:0] q
);

assign q = d & {8{count[2]}} & {8{count[1]}} & {8{count[0]}};

endmodule

// top_module.v
module top_module (
    input clk,
    input reset,
    input [7:0] d,
    output [7:0] q
);

wire [2:0] count;
wire [7:0] dff_out;

dff_reset dff_inst (
    .clk(clk),
    .reset(count[2]),
    .d(d),
    .q(dff_out)
);

counter counter_inst (
    .clk(clk),
    .reset(reset),
    .count(count)
);

and_module and_inst (
    .d(dff_out),
    .count(count),
    .q(q)
);

endmodule
