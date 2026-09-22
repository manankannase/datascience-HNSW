`timescale 1ns/1ps
module SimplifiedMasterController(
    input wire clk,
    input wire reset,
    input wire start,
    input wire [7:0] A00, A01, A02,
    input wire [7:0] A10, A11, A12,
    input wire [7:0] A20, A21, A22,
    input wire [7:0] B00, B01, B02,
    input wire [7:0] B10, B11, B12,
    input wire [7:0] B20, B21, B22,
    output reg [7:0] a1, a2, a3,
    output reg [7:0] b1, b2, b3,
    output reg done,
    output reg clear
);
    reg [2:0] cycle;
    localparam IDLE = 2'd0,
               CLEAR = 2'd1,
               FEED = 2'd2,
               COMPLETE = 2'd3;
    reg [1:0] state;
    
    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            cycle <= 0;
            a1 <= 0; a2 <= 0; a3 <= 0;
            b1 <= 0; b2 <= 0; b3 <= 0;
            done <= 0; clear <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0; clear <= 0;
                    if (start) begin
                        state <= CLEAR;
                        cycle <= 0;
                    end
                end
                CLEAR: begin
                    clear <= 1;
                    state <= FEED;
                    cycle <= 0;
                end
                FEED: begin
                    clear <= 0;
                    case (cycle)
                        3'd0: begin
                            a1 <= A00; a2 <= 8'd0; a3 <= 8'd0;
                            b1 <= B00; b2 <= 8'd0; b3 <= 8'd0;
                        end
                        3'd1: begin
                            a1 <= A01; a2 <= A10; a3 <= 8'd0;
                            b1 <= B10; b2 <= B01; b3 <= 8'd0;
                        end
                        3'd2: begin
                            a1 <= A02; a2 <= A11; a3 <= A20;
                            b1 <= B20; b2 <= B11; b3 <= B02;
                        end
                        3'd3: begin
                            a1 <= 8'd0; a2 <= A12; a3 <= A21;
                            b1 <= 8'd0; b2 <= B21; b3 <= B12;
                        end
                        3'd4: begin
                            a1 <= 8'd0; a2 <= 8'd0; a3 <= A22;
                            b1 <= 8'd0; b2 <= 8'd0; b3 <= B22;
                        end
                        3'd5, 3'd6: begin
                            a1 <= 8'd0; a2 <= 8'd0; a3 <= 8'd0;
                            b1 <= 8'd0; b2 <= 8'd0; b3 <= 8'd0;
                        end
                    endcase
                    if (cycle == 3'd7)
                        state <= COMPLETE;
                    else
                        cycle <= cycle + 1;
                end
                COMPLETE: begin
                    done <= 1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule


