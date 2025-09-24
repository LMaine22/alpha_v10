# Single Ticker Mode Documentation

## Overview
Single ticker mode allows you to focus discovery on generating trading setups for just one ticker, while still using all tickers in the universe for signal generation. This is useful for:

- Deep analysis of a specific ticker's trading opportunities
- Reducing computational complexity for faster experimentation 
- Testing strategies on a single high-interest ticker

## How It Works

When single ticker mode is enabled:
1. **Setup Generation**: Only the specified ticker will be used for creating trading setups
2. **Signal Discovery**: ALL tickers (tradable + macro) are still used to generate signals and features
3. **Cross-Asset Signals**: You can still discover setups that use signals from other tickers to trade the target ticker

## Configuration

In `alpha_discovery/config.py`, set the `single_ticker_mode` parameter:

```python
class DataConfig(BaseModel):
    # ... other fields ...
    
    # Single ticker mode: if set, only this ticker will be used for trading setups
    # All other tradable_tickers will be treated like macro_tickers (signals only, no setups)
    single_ticker_mode: Optional[str] = None  # Example: 'AAPL US Equity' to focus on AAPL only
```

## Usage Examples

### Example 1: Focus on AAPL
```python
# In config.py
single_ticker_mode: Optional[str] = 'AAPL US Equity'
```
Result: All GA individuals will be AAPL setups, but can use signals from MSFT, TSLA, SPY, RTY Index, etc.

### Example 2: Focus on TSLA  
```python
# In config.py
single_ticker_mode: Optional[str] = 'TSLA US Equity'
```
Result: Only TSLA trading setups, but leveraging the full signal universe.

### Example 3: Normal Multi-Ticker Mode
```python
# In config.py  
single_ticker_mode: Optional[str] = None
```
Result: Normal behavior - setups can be generated for any ticker in `tradable_tickers`.

## Implementation Details

- **Population Initialization**: Only creates individuals with the specified ticker
- **Crossover**: Offspring automatically inherit the single ticker (no ticker mixing)
- **Mutation**: Ticker mutations are disabled; only signal mutations occur
- **Error Handling**: Validates that the specified ticker exists in `tradable_tickers`

## Benefits

1. **Focused Analysis**: Concentrate computational resources on one ticker
2. **Cross-Asset Intelligence**: Still leverage signals from the entire universe
3. **Faster Iteration**: Smaller search space means faster discovery cycles
4. **Specialized Strategies**: Develop ticker-specific trading approaches

## Best Practices

- Choose liquid, actively traded tickers for single ticker mode
- Consider using tickers with interesting cross-asset relationships
- Use during development/testing phases, then expand to multi-ticker for production
- Monitor that the chosen ticker has sufficient signal coverage and support