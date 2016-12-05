<flow>
    <moving_average window="5">
        <feature>value(t+1)</feature>
        <feature>value(t+2)</feature>
        <feature>value(t+3)</feature>
        <feature>value(t+4)</feature>
        <feature>value(t+5)</feature>
    </moving_average>
    <moving_standard_deviation window="5">
        <feature>value(t+1)</feature>
        <feature>value(t+2)</feature>
        <feature>value(t+3)</feature>
        <feature>value(t+4)</feature>
        <feature>value(t+5)</feature>
    </moving_standard_deviation>
</flow>