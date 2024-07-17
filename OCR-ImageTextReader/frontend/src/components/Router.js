import React from "react";
import { Route, Switch } from "react-router-dom";
import AppFields from "./AppFields";

const Routes = props => {
  return (
    <Switch>
      <Route exact path="/home" component={AppFields} />
    </Switch>
  );
};

export default Routes;